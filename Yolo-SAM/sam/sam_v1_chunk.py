import os
import cv2
import torch
from ultralytics import SAM
from tqdm import tqdm
import argparse
import time
import multiprocessing as mp

RESIZE = 1          # 1, 1/2, 1/4
SKIP_FRAMES = 300   # 1, 2, 5, 10, 30
CHUNK_SIZE = 10     # Number of frames to process in each chunk

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and segment video frames.")
    parser.add_argument('input', type=str, help="Path to the input video file.")
    parser.add_argument('output', type=str, nargs='?', help="Path to the output video file.")
    return parser.parse_args()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def process_chunk(chunk, seg_frame_size, device):
    model = SAM("sam2_b.pt").to(device)
    processed_frames = []
    for frame in chunk:
        try:
            result = model(frame)
            seg_frame = result[0].plot()
            seg_frame = cv2.resize(seg_frame, seg_frame_size)
            processed_frames.append(seg_frame)
        except AttributeError as e:
            print(f"Error: SAM model doesn't have the expected method or attribute. Details: {str(e)}")
        except IndexError as e:
            print(f"Error: Unexpected result structure from SAM model. Details: {str(e)}")
        except Exception as e:
            print(f"Error during model inference or result processing: {str(e)}")
    return processed_frames

def process_chunk_wrapper(args):
    chunk, seg_frame_size, device = args
    return process_chunk(chunk, seg_frame_size, device)

def main():
    args = parse_arguments()
    input_vid = args.input
    output_vid = args.output if args.output else "seg_" + os.path.basename(input_vid)

    device = get_device()
    print(f"> Using device: {device}")

    # Raw video properties
    cap = cv2.VideoCapture(input_vid)  # type: ignore
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    print(f"Raw video: {input_vid}  |  {total_frames} frames  |  {frame_size} size  |  {fps:.2f} fps")

    # Segmented video properties
    seg_width, seg_height = int(width * RESIZE), int(height * RESIZE)
    seg_frame_size = (seg_width, seg_height)
    seg_fps = fps / SKIP_FRAMES
    print(f"Seg video: {output_vid}  |  {int(total_frames/SKIP_FRAMES + 1)} frames  |  {seg_frame_size} size  |  {seg_fps:.2f} fps")

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_vid, fourcc, float(fps/SKIP_FRAMES), seg_frame_size, isColor=True)

    # Read frames
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i % SKIP_FRAMES == 0:
            frames.append(frame)
    cap.release()

    # Process video
    total_start_time = time.time()
    chunks = [frames[i:i + CHUNK_SIZE] for i in range(0, len(frames), CHUNK_SIZE)]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        chunk_args = [(chunk, seg_frame_size, device) for chunk in chunks]
        results = []
        for i, result in enumerate(tqdm(pool.imap(process_chunk_wrapper, chunk_args), total=len(chunks), desc="Processing chunks")):
            chunk_time = time.time() - total_start_time
            print(f"Chunk {i+1}/{len(chunks)} processed in {chunk_time:.2f}s")
            results.append(result)

    for chunk_result in results:
        for seg_frame in chunk_result:
            out.write(seg_frame)

    out.release()
    cv2.destroyAllWindows()

    total_processing_time = time.time() - total_start_time
    print(f">> Total processing time for the entire video: {total_processing_time:.2f}s")

    if os.path.exists(output_vid) and os.path.getsize(output_vid) > 0:
        print(f"Output video created successfully: {output_vid}")

        # Segmented video properties
        seg_cap = cv2.VideoCapture(output_vid)  # type: ignore
        if seg_cap.isOpened():
            seg_fps = seg_cap.get(cv2.CAP_PROP_FPS)
            seg_total_frames = int(seg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Seg video: {output_vid} | {seg_total_frames} frames | {seg_frame_size} size | {seg_fps} fps")
            seg_cap.release()
        else:
            print(f"Error: Could not open the output video file: {output_vid}")
    else:
        print(f"Error: Output video file not created or empty: {output_vid}")
        return

if __name__ == '__main__':
    main()
