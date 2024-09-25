import os
import cv2
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO, SAM
import psutil
from typing import List, Tuple

RESIZE = 1           # 1, 1/2, 1/4
SKIP_FRAMES = 30     # Set to 1 to process all frames
MAX_FRAMES = None    # Set to None to process all frames

# Get device
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Estimate optimal chunk size
def estimate_optimal_chunk_size(frame_size, total_frames, skip_frames):
    frame_memory = np.prod(frame_size) * 3 * 4      # width * height * channels * bytes_per_float
    available_memory = psutil.virtual_memory().available * 0.75
    max_frames_in_memory = available_memory // frame_memory
    effective_frames = total_frames // skip_frames
    chunk_size = min(10, max_frames_in_memory, effective_frames)
    return int(chunk_size)

# Parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Object segmentation using YOLO and SAM")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_video", nargs="?", default=None, help="Path to the output video file (optional)")
    parser.add_argument("-d", "--detect", nargs="+", help="Objects to detect and segment")
    args = parser.parse_args()
    if not os.path.isfile(args.input_video):
        parser.error(f"Input video file '{args.input_video}' does not exist")
    return args

# Process a chunk of frames.
def process_chunk(frames: List[np.ndarray], model: YOLO, seg_frame_size: Tuple[int, int], start_frame_idx: int, progress_bar: tqdm) -> Tuple[List[np.ndarray], List[float], List[int]]:
    processed_frames, processing_times, frame_numbers = [], [], []
    for i, frame in enumerate(frames):
        if frame.shape[:2][::-1] != seg_frame_size:
            frame = cv2.resize(frame, seg_frame_size)
        start_time = time.time()
        results = model(frame, stream=True)
        for result in results:
            seg_frame = result.plot(boxes=False)        # seg_frame = result.plot()
            processing_time = time.time() - start_time
            processed_frames.append(seg_frame)
            processing_times.append(processing_time)
            frame_number = start_frame_idx + i * SKIP_FRAMES
            frame_numbers.append(frame_number)
            tqdm.write(f"> Frame {frame_number}: Processing time {processing_time:.2f}s")
            progress_bar.update(1)
            break
    return processed_frames, processing_times, frame_numbers

# Filter detections based on specified objects.
def filter_detections(detections: torch.Tensor, yolo_model: YOLO, objects_to_detect: List[str]) -> List[torch.Tensor]:
    filtered_detections = []
    for obj in objects_to_detect:
        obj_detections = [d for d in detections if yolo_model.names[int(d[5])] == obj]
        if obj_detections:
            largest_detection = max(obj_detections, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
            filtered_detections.append(largest_detection)
    return filtered_detections

# Apply segmentation to the frame.
def apply_segmentation(frame: np.ndarray, sam_model: SAM, boxes: np.ndarray) -> np.ndarray:
    labels = np.ones(len(boxes))
    sam_results = sam_model(frame, bboxes=boxes, points=None, labels=labels)
    
    if sam_results[0].masks is not None:
        masks = sam_results[0].masks.data.cpu().numpy()
        seg_frame = frame.copy()
        for mask in masks:
            color = np.random.randint(0, 255, size=3, dtype=np.uint8)
            seg_frame[mask] = seg_frame[mask] * 0.5 + color * 0.5
        return seg_frame.astype(np.uint8)
    return frame

# Process the entire video.
def process_video(input_vid: str, output_vid: str = None, objects_to_detect: List[str] = None) -> None:
    device = get_device()
    print(f"> Using device: {device}")

    yolo_model = YOLO("model/yolov8n.pt").to(device)
    sam_model = SAM("model/sam2_b.pt").to(device)

    cap = cv2.VideoCapture(input_vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    print(f"> Raw video: {input_vid}  |  {total_frames} frames  |  {frame_size} size  |  {fps:.2f} fps")

    seg_frame_size = (int(width * RESIZE), int(height * RESIZE))
    seg_fps = fps / SKIP_FRAMES

    if output_vid is None:
        input_name = os.path.splitext(os.path.basename(input_vid))[0]
        output_vid = f"seg_{input_name}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_vid, fourcc, seg_fps, seg_frame_size)

    frames_processed = 0
    total_start_time = time.time()
    progress_bar = tqdm(total=total_frames // SKIP_FRAMES, desc="Processing frames")

    while True:
        for _ in range(SKIP_FRAMES):
            ret, frame = cap.read()
            if not ret:
                break

        if not ret:
            break

        frame = cv2.resize(frame, seg_frame_size) if frame.shape[:2][::-1] != seg_frame_size else frame

        yolo_result = yolo_model(frame)[0]
        detections = yolo_result.boxes.data
        
        if objects_to_detect:
            detections = filter_detections(detections, yolo_model, objects_to_detect)

        if detections:
            detections = torch.stack(detections) if isinstance(detections[0], torch.Tensor) else torch.tensor(detections)
            boxes = detections[:, :4].cpu().numpy()
            seg_frame = apply_segmentation(frame, sam_model, boxes)
        else:
            seg_frame = frame

        if seg_frame.shape[:2][::-1] != frame_size:
            seg_frame = cv2.resize(seg_frame, frame_size)

        out.write(seg_frame)

        cv2.imshow('Segmentation', seg_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frames_processed += 1
        progress_bar.update(1)

        if MAX_FRAMES and frames_processed >= MAX_FRAMES:
            break

    progress_bar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_processing_time = time.time() - total_start_time
    print(f"> Total processing time: {total_processing_time:.2f}s")
    print(f"> Average time per processed frame: {total_processing_time/frames_processed:.2f}s")
    print(f"> Seg video: {output_vid}  |  {frames_processed} frames  |  {seg_frame_size} size  |  {seg_fps:.2f} fps")

def main():
    args = parse_arguments()
    process_video(args.input_video, args.output_video, args.detect)

if __name__ == "__main__":
    main()
