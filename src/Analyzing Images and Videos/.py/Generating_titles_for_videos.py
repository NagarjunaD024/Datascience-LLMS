import argparse
import openai
import pandas as pd
import time
from IPython.display import Markdown, display, Image
from google.colab import files
import base64
import os
import requests
import shutil
import cv2

from google.colab import drive, userdata
from openai import OpenAI
drive.mount('/content/drive')

try:
    API_KEY = userdata.get('OPENAI_API_KEY')
    if not API_KEY:
        raise ValueError("API key not found. Please add it to Colab Secrets.")
except ImportError:
    # Fallback for environments other than Colab, though not recommended
    API_KEY = os.environ.get("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("API_KEY environment variable not set.")

print(API_KEY)

def extract_frames(video_path):
    """ Extracts frames from a video.

    Args:
        video_path: path to video file.

    Returns:
        list of first ten video frames.
    """
    video = cv2.VideoCapture(video_path)
    frames = []
    while video.isOpened() and len(frames) <= 10:
        success, frame = video.read()
        if not success:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        encoded = base64.b64encode(buffer)
        frame = encoded.decode('utf-8')
        frames += [frame]

    video.release()
    return frames

def create_prompt(frames):
    """ Create prompt to generate title for video.

    Args:
        frames: frames of video.

    Returns:
        prompt containing multimodal data (as list).
    """
    prompt = ['Generate a concise title for the video.']
    for frame in frames[:10]:
        element = {'image':frame, 'resize':768}
        prompt += [element]
    return prompt

client = openai.OpenAI(api_key= userdata.get('OPENAI_API_KEY'))

def call_llm(prompt):
    """ Query large language model and return answer.

    Args:
        prompt: input prompt for language model.

    Returns:
        Answer by language model.
    """
    for nr_retries in range(1, 4):
        try:
            response = client.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {'role':'user', 'content':prompt}
                    ]
                )
            return response.choices[0].message.content
        except:
            time.sleep(nr_retries * 2)
    raise Exception('Cannot query OpenAI model!')

if __name__ == '__main__':
  """
  for Script Invoking from command line with argparse Ex- cd Generating titles for videos.py "/content/drive/MyDrive/Colab Notebooks/DataScience+GPT/Data/image and video/cars.mp4"
  """
    parser = argparse.ArgumentParser()
    parser.add_argument('videopath', type=str, help='Path of video file')
    args = parser.parse_args()

    frames = extract_frames(args.videopath)
    prompt = create_prompt(frames)
    title = call_llm(prompt)
    print(title)