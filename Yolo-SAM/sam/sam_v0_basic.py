import os
import cv2
import torch
from ultralytics import SAM
from tqdm import tqdm
import argparse
import time

RESIZE = 1           # 1, 1/2, 1/4
SKIP_FRAMES = 300    # 1, 2, 5, 10, 30

parser = argparse.ArgumentParser(description="Process and segment video frames.")
parser.add_argument('input', type=str, help="Path to the input video file.")
parser.add_argument('output', type=str, nargs='?', help="Path to the output video file.")
args = parser.parse_args()
input_vid = args.input
output_vid = args.output if args.output else "seg_" + os.path.basename(input_vid)

# device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")  # on Mac, but doesn't work !
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       # on Linux/Windows w/ Nvidia GPU
model = SAM("sam2_b.pt")
model.to(device)
gpu = device.type == 'cuda'
print(f"> Using device: {device}")

# Raw video properties
cap = cv2.VideoCapture(input_vid)  # type: ignore
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
print(f"Raw video: {input_vid}  |  {total_frames} frames  |  {frame_size} size  |  {fps:.2f} fps")
print("    --- Resizing to :", RESIZE, "... Skip frames :", SKIP_FRAMES, "---")

# Segmented video properties
seg_width, seg_height = int(width * RESIZE), int(height * RESIZE)
seg_frame_size = (seg_width, seg_height)
seg_fps = fps / SKIP_FRAMES
print(f"Seg video: {output_vid}  |  {int(total_frames/SKIP_FRAMES + 1)} frames  |  {seg_frame_size} size  |  {seg_fps:.2f} fps")

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter()
if not out.open(output_vid, fourcc, float(fps/SKIP_FRAMES), seg_frame_size, isColor=True):
    print(f"Error: Could not create output video file {output_vid}")
    exit()

total_start_time = time.time()
frame_counter = 0

try:
    frames_to_process = range(0, total_frames, SKIP_FRAMES)
    print(f"> Number of frames to process: {len(frames_to_process)}")
    for frame_counter in tqdm(frames_to_process, desc="Processing ...", dynamic_ncols=True):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.resize(frame, frame_size)
        frame_start_time = time.time()

        try:
            results = model(frame, stream=True)
            for result in results:
                seg_frame = result.plot()
                seg_frame = cv2.resize(seg_frame, seg_frame_size)

                # cv2.imshow('Raw Video', frame)
                cv2.imshow('Seg Video', seg_frame)      # cv2.imshow('Seg Video', seg_frame_bgr)
                out.write(seg_frame)                    # out.write(seg_frame_bgr)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting ...")
                    raise KeyboardInterrupt

        except Exception as e:
            print(f"Error during model inference or result processing: {str(e)}")
            continue

except KeyboardInterrupt:
    print(f"Interrupted at frame {frame_counter}")

finally:
    out.release()
    cv2.destroyAllWindows()
    cap.release()

    total_processing_time = time.time() - total_start_time
    print(f">> Total processing time for the entire video: {total_processing_time:.2f}s")

    # Segmented video properties
    seg_cap = cv2.VideoCapture(output_vid)  # type: ignore
    if seg_cap.isOpened():
        seg_fps = seg_cap.get(cv2.CAP_PROP_FPS)
        seg_total_frames = int(seg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Seg video: {output_vid} | {seg_total_frames} frames | {seg_frame_size} size | {seg_fps} fps")
        seg_cap.release()
