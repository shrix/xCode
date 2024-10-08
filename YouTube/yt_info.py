# pip install google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib opencv-python pytesseract pytube

import os
import re
import cv2
import yt_dlp
import requests
import tldextract
import pytesseract
from googleapiclient.discovery import build
from concurrent.futures import ThreadPoolExecutor, as_completed

YOUTUBE_API_KEY = os.environ['YOUTUBE_API_KEY']
MAX_RESULTS = 500
SKIP_FRAMES = 8

if not os.path.exists('info'):
    os.makedirs('info')


# Function to extract URLs from text
def extract_urls(text):
    return re.findall(r'(?:http[s]?://)?(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(?:/[^\s]*)?', text)
                    # r'\b(?:https?://)?(?:www\.)?[a-zA-Z0-9.-]+\.(?:com|org|net|gov|edu|mil|int|biz|info|name|museum|coop|aero|[a-z]{2})\b'

# Function to find URLs in an image using OCR.
# Get the description and URLs present within a YouTube video.
def get_youtube_desc_urls(video_id):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    request = youtube.videos().list(
        part="snippet",
        id=video_id
    )
    response = request.execute()
    if not response['items']:
        print(f"No video found with ID {video_id}")
        return None, None
    description = response['items'][0]['snippet']['description']
    urls = extract_urls(description)
    return description, urls


### YOUTUBE VIDEO EXTRACTION FUNCTIONS ###
# Function to capture frames from a video at the specified interval (in seconds).
def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps * SKIP_FRAMES)
    try:
        success, image = cap.read()
        frame_count = 0
        while success:
            if frame_count % frame_skip == 0:
                yield image
            success, image = cap.read()
            frame_count += 1
    finally:
        cap.release()

def find_urls(image):           # find_text_urls
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    urls = extract_urls(text)
    return urls                 # could return 'text' in the video as well

def process_frame(frame):
    urls = find_urls(frame)
    return urls

# Function to download a YouTube video, capture frames, and find URLs in the frames.
def get_youtube_vidurls(video_id):
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    print(f'Downloading video {video_id} ...')
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]', # Download best video and audio formats
        'outtmpl': f'{video_id}.mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    # Capture frames and find URLs in the frames
    urls = []
    print(f'Processing video {video_id} ...', end="")
    try:
        with ThreadPoolExecutor() as executor:
            frames = list(get_frames(f'{video_id}.mp4'))
            total_frames = len(frames)
            results = executor.map(process_frame, frames)
            for i, frame_urls in enumerate(results):
                urls += frame_urls
                print(f'\rProcessing video {video_id} ... {int((i+1)/total_frames*100)}%', end="")
    except Exception as e:
        print(f'Error processing a frame: {e}')
    finally:
        os.remove(f'{video_id}.mp4')            # Remove video file
    print()
    return urls


### YOUTUBE CHANNEL EXTRACTION FUNCTIONS ###
# Function to search for a YouTube channel and get the latest video.
def youtube_chan_search(channel_id):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(
        part="snippet",
        maxResults=1,
        q=channel_id,
        type="channel"
    )
    response = request.execute()
    channel_id = response['items'][0]['id']['channelId']
    request = youtube.search().list(                    # Get all videos from the channel
        part="snippet",
        maxResults=MAX_RESULTS,
        channelId=channel_id,
        type="video"
    )
    return youtube, request.execute()

# Function to get the description and URLs from a YouTube channel.
def get_youtube_chan_desc_urls(channel_id):
    youtube, response = youtube_chan_search(channel_id)
    urls = []
    for item in response['items']:
        video_id = item['id']['videoId']
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()
        description = response['items'][0]['snippet']['description']
        urls += extract_urls(description)
    return description, urls

# Function to download a YouTube video, capture frames, and find URLs in the frames.
def get_youtube_chan_vidsurls(channel_id):
    youtube, response = youtube_chan_search(channel_id)
    urls = []
    video_ids = [item['id']['videoId'] for item in response['items']]
    # Use a ThreadPoolExecutor to download and process videos in parallel
    with ThreadPoolExecutor() as executor:
        future_to_video = {executor.submit(get_youtube_vidurls, video_id): video_id for video_id in video_ids}
        for future in as_completed(future_to_video):
            vid_urls = future.result()
            urls += vid_urls
    return urls


### RUN THE CODE ###
# 1.Get the link of a YouTube video from the user.
video_link = input('Enter the link of a YouTube video: ')
if 'youtu.be' in video_link:
    video_id = video_link.split('/')[-1]
else:
    video_id = video_link.split('=')[-1]
# 2.Get the URLs from the description of the video.
response = input('\nDo you want to extract the text & URLs from the description (Y/N)?: ')
if response.lower() == 'y':
    description, urls = get_youtube_desc_urls(video_id)
    unique_urls = set(urls)
    print(f"\n>> Description: ...\n\n{description}")
    with open('info/' + video_id + '_desc_text.txt', 'w') as f:
        f.write(description)
    print("\n>> URLs in description: ...\n")
    with open('info/' + video_id + '_desc_urls.txt', 'w') as f:
        for url in unique_urls:
            print(url)
            f.write(url + '\n')
# 3.Get the URLs from within the video.
response = input('\nDo you want to extract the text & URLs from within the video (Y/N)?: ')
if response.lower() == 'y':
    urls = get_youtube_vidurls(video_id)
    unique_urls = set(urls)
    print("\n>> URLs in video: ...\n")
    with open('info/' + video_id + '_vid_urls.txt', 'w') as f:
        for url in unique_urls:
            print(url)
            f.write(url + '\n')


# # 1.Get the name of the YouTube channel from the user.
# channel_name = input('Enter name of the YouTube channel: ')
# # 2.Get the URLs from the description of the latest video from the channel.
# response = input('\nDo you want to extract the text & URLs from their description (Y/N)?: ')
# if response.lower() == 'y':
#     description, urls = get_youtube_chan_desc_urls(channel_name)
#     unique_urls = set(urls)
#     print(f"\n>> Description: ...\n\n{description}")
#     with open('info/' + channel_name + '_desc_text.txt', 'w') as f:
#         f.write(description)
#     print("\n>> URLs in description: ...\n")
#     with open('info/' + channel_name + '_desc_urls.txt', 'w') as f:
#         for url in unique_urls:
#             print(url)
#             f.write(url + '\n')
# # 3.Get the URLs from within the videos of the channel.
# response = input('\nDo you want to extract the text & URLs from within the videos (Y/N)?: ')
# if response.lower() == 'y':
#     urls = get_youtube_chan_vidsurls(channel_name)
#     unique_urls = set(urls)
#     print("\n>> URLs in video: ...\n")
#     with open('info/' + channel_name + '_vid_urls.txt', 'w') as f:
#         for url in unique_urls:
#             print(url)
#             f.write(url + '\n')
