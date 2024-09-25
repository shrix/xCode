import os
import cv2
import torch
from ultralytics import SAM
from tqdm import tqdm
import argparse
import time
from multiprocessing import Pool, cpu_count, freeze_support
from functools import partial
import numpy as np

RESIZE = 1           # 1, 1/2, 1/4
SKIP_FRAMES = 30     # 1, 2, 5, 10, 30
BATCH_SIZE = 4       # Based on GPU memory
MAX_FRAMES = 3       # Max frames to process

# Process a single frame using the CPU.
def process_frame_cpu(frame_data, model, seg_frame_size):
    frame, frame_number = frame_data
    start_time = time.time()
    if frame.shape[:2][::-1] != seg_frame_size:
        frame = cv2.resize(frame, seg_frame_size)
    results = model(frame, stream=True)
    for result in results:
        seg_frame = result.plot()
        processing_time = time.time() - start_time
        return seg_frame, processing_time
    return None, time.time() - start_time

# Process a batch of frames using the CPU.
def process_batch_cpu(batch, model, seg_frame_size):
    results = []
    processing_times = []
    print(f"> CPU Batch: Processing {len(batch)} frames")
    for i, frame_data in enumerate(batch):
        result, proc_time = process_frame_cpu(frame_data, model, seg_frame_size)
        results.append(result)
        processing_times.append(proc_time)
        print(f"> CPU Frame {i+1}/{len(batch)} processed in {proc_time:.2f}s")
    return results, processing_times

# Process a batch of frames using the GPU.
def process_batch_gpu(batch, model, seg_frame_size):
    frames, frame_numbers = zip(*batch)
    frames = [cv2.resize(frame, seg_frame_size) for frame in frames]
    print(f"> GPU Batch: Processing {len(batch)} frames")
    start_time = time.time()
    results = model(frames, stream=True)
    seg_frames = [result.plot() for result in results]
    processing_time = time.time() - start_time
    print(f"> GPU Batch processed in {processing_time:.2f}s")
    return seg_frames, list(frame_numbers), [processing_time / len(frames)] * len(frames)

# Process video chunks.
def process_video_chunks(input_vid, skip_frames, batch_size, max_frames):
    cap = cv2.VideoCapture(input_vid)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_vid}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    frames_to_process = []

    # Calculate which frames we actually want to process
    while frame_index < total_frames and len(frames_to_process) < max_frames:
        frames_to_process.append(frame_index)
        frame_index += skip_frames + 1

    for start_idx in range(0, len(frames_to_process), batch_size):
        batch = []
        end_idx = min(start_idx + batch_size, len(frames_to_process))
        
        for frame_idx in frames_to_process[start_idx:end_idx]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"> Reached end of video at frame {frame_idx}")
                break
            batch.append((frame, frame_idx))

        if not batch:
            print("> No more frames to process")
            break

        print(f"\n> Yielding batch with {len(batch)} frames - Frames to process: {len(frames_to_process)}")
        yield batch

    cap.release()

# Main function to process video.
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

    # Initialize the model and device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAM("sam2_b.pt").to(device)
    gpu = device.type == 'cuda'
    print(f"> Using device: {device}")

    # Raw video properties
    cap = cv2.VideoCapture(input_vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    print(f"> Raw video: {input_vid}  |  {total_frames} frames  |  {frame_size} size  |  {fps:.2f} fps")

    # Segmented video properties
    seg_width, seg_height = int(width * RESIZE), int(height * RESIZE)
    seg_frame_size = (seg_width, seg_height)
    seg_fps = fps / SKIP_FRAMES

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_vid, fourcc, float(fps/SKIP_FRAMES), seg_frame_size, isColor=True)
    if not out.isOpened():
        print(f"> Error: Could not create output video file {output_vid}")
        return

    total_start_time = time.time()
    total_processing_time = 0
    frames_processed = 0

    try:
        for batch in tqdm(process_video_chunks(input_vid, SKIP_FRAMES, BATCH_SIZE, MAX_FRAMES), desc="> Processing chunks"):
            batch_start_time = time.time()
            
            # Trim the batch to MAX_FRAMES if needed
            frames_left = MAX_FRAMES - frames_processed
            if frames_left <= 0:
                break
            batch = batch[:frames_left]
            
            if gpu:
                print(f"> Processing batch of {len(batch)} frames on GPU")
                seg_frames, frame_numbers, proc_times = process_batch_gpu(batch, model, seg_frame_size)
            else:
                print(f"> Processing batch of {len(batch)} frames on CPU")
                seg_frames, proc_times = process_batch_cpu(batch, model, seg_frame_size)
                frame_numbers = [frame_num for _, frame_num in batch]
            
            for seg_frame, frame_number, proc_time in zip(seg_frames, frame_numbers, proc_times):
                if seg_frame is not None:
                    out.write(seg_frame)
                    # cv2.imshow('Seg Video', seg_frame)
                    # print(f"> Frame {frame_number} processed in {proc_time:.2f}s and written to output")
                    total_processing_time += proc_time
                    frames_processed += 1
            
            batch_processing_time = time.time() - batch_start_time
            print(f"\n> Batch processed in {batch_processing_time:.2f}s")
            print(f"> Frames processed so far: {frames_processed}/{MAX_FRAMES}")

            # if frames_processed >= MAX_FRAMES or cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("> Reached max frames or quit signal received")
            #     break

    except KeyboardInterrupt:
        print("> Interrupted by user")

    finally:
        out.release()
        cv2.destroyAllWindows()
        if 'cap' in locals():
            cap.release()

        total_elapsed_time = time.time() - total_start_time
        print(f"> Total processing time for all frames: {total_processing_time:.2f}s")
        print(f"> Total elapsed time (including I/O): {total_elapsed_time:.2f}s")
        if frames_processed > 0:
            print(f"> Average processing time per frame: {total_processing_time / frames_processed:.2f}s")

        # Segmented video properties
        seg_cap = cv2.VideoCapture(output_vid)
        if seg_cap.isOpened():
            seg_fps = seg_cap.get(cv2.CAP_PROP_FPS)
            seg_total_frames = int(seg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            expected_frames = min(int(total_frames/SKIP_FRAMES + 1), MAX_FRAMES)
            print(f"> Seg video: {output_vid}  |  {expected_frames} frames  |  {seg_frame_size} size  |  {seg_fps:.2f} fps")
            seg_cap.release()

        # Count actual frames in output video
        def count_frames(video_path):
            cap = cv2.VideoCapture(video_path)
            count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1
            cap.release()
            return count

        actual_frame_count = count_frames(output_vid)
        print(f"> Actual frames in output video: {actual_frame_count}")

if __name__ == '__main__':
    freeze_support()
    main()
