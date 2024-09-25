import os
import cv2
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import SAM
import psutil

RESIZE = 1           # 1, 1/2, 1/4
SKIP_FRAMES = 30     # 1, 2, 5, 10, 30
MAX_FRAMES = 3       # Set to None to process all frames

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def estimate_optimal_chunk_size(frame_size, total_frames, skip_frames):
    frame_memory = np.prod(frame_size) * 3 * 4      # width * height * channels * bytes_per_float
    available_memory = psutil.virtual_memory().available * 0.75
    max_frames_in_memory = available_memory // frame_memory
    effective_frames = total_frames // skip_frames
    chunk_size = min(10, max_frames_in_memory, effective_frames)
    return int(chunk_size)

def process_chunk(chunk, model, seg_frame_size, start_frame_number, progress_bar):
    processed_frames = []
    processing_times = []
    frame_numbers = []
    for i, frame in enumerate(chunk):
        if frame.shape[:2][::-1] != seg_frame_size:
            frame = cv2.resize(frame, seg_frame_size)
        start_time = time.time()
        results = model(frame, stream=True)
        for result in results:
            seg_frame = result.plot()
            processing_time = time.time() - start_time
            processed_frames.append(seg_frame)
            processing_times.append(processing_time)
            frame_number = start_frame_number + i * SKIP_FRAMES
            frame_numbers.append(frame_number)
            tqdm.write(f"> Frame {frame_number}: Processing time {processing_time:.2f}s")
            progress_bar.update(1)
            break
    return processed_frames, processing_times, frame_numbers

def process_video(input_vid, output_vid):
    device = get_device()
    print(f"> Using device: {device}")

    model = SAM("sam2_b.pt").to(device)

    # Get video info
    cap = cv2.VideoCapture(input_vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    print(f"> Raw video: {input_vid}  |  {total_frames} frames  |  {frame_size} size  |  {fps:.2f} fps")

    # Set segment frame size and fps
    seg_frame_size = (width, height)
    seg_fps = fps / SKIP_FRAMES

    # Estimate optimal chunk size
    chunk_size = estimate_optimal_chunk_size(frame_size, total_frames, SKIP_FRAMES)
    print(f"> Estimated optimal chunk size: {chunk_size}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_vid, fourcc, float(fps/SKIP_FRAMES), seg_frame_size, isColor=True)

    total_start_time = time.time()
    total_processing_time = 0
    frames_processed = 0

    # Get frames to process
    frames_to_process = list(range(0, total_frames, SKIP_FRAMES))[:MAX_FRAMES]
    total_frames_to_process = len(frames_to_process)
    print(f"> Processing {total_frames_to_process} frames ...")

    progress_bar = tqdm(total=total_frames_to_process, desc="Processing frames")

    for frame_idx in frames_to_process:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process single frame
        seg_frames, proc_times, frame_numbers = process_chunk([frame], model, seg_frame_size, frame_idx, progress_bar)
        
        # Write processed frame
        if seg_frames[0] is not None:
            out.write(seg_frames[0])
            total_processing_time += proc_times[0]
            frames_processed += 1

    progress_bar.close()
    cap.release()
    out.release()

    total_elapsed_time = time.time() - total_start_time
    print(f"> Total processing time for all frames: {total_processing_time:.2f}s")
    print(f"> Total elapsed time (including I/O): {total_elapsed_time:.2f}s")
    if frames_processed > 0:
        print(f"> Average processing time per frame: {total_processing_time / frames_processed:.2f}s")

    print(f"> Seg video: {output_vid}  |  {frames_processed} frames  |  {seg_frame_size} size  |  {seg_fps:.2f} fps")

def main():
    parser = argparse.ArgumentParser(description="Process and segment video frames.")
    parser.add_argument('input', type=str, help="Path to the input video file.")
    parser.add_argument('output', type=str, nargs='?', help="Path to the output video file.")
    args = parser.parse_args()

    input_vid = args.input
    output_vid = args.output if args.output else "seg_" + os.path.basename(input_vid)

    if not os.path.exists(input_vid):
        print(f"> Error: Input video file '{input_vid}' does not exist.")
        return

    process_video(input_vid, output_vid)

if __name__ == '__main__':
    main()
