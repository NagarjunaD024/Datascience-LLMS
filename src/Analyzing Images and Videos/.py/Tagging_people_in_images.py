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

from google.colab import drive, userdata
from openai import OpenAI
drive.mount('/content/drive')

def load_images(in_dir):
    """ Loads images from a directory.

    Args:
        in_dir: path of input directory.

    Returns:
        directory mapping file names to PNG images.
    """
    name_to_image = {}
    file_names = os.listdir(in_dir)
    for file_name in file_names:
        if file_name.endswith('.png'):
            image_path = os.path.join(in_dir, file_name)
            with open(image_path, 'rb') as image_file:
                encoded = base64.b64encode(image_file.read())
                image = encoded.decode('utf-8')
                name_to_image[file_name] = image

    return name_to_image

def create_prompt(person_image, image_to_label):
    """ Create prompt to compare images.

    Args:
        person_image: image showing a person.
        image_to_label: image to assign to a label.

    Returns:
        prompt to verify if the same person appears in both images.
    """
    task = {'type':'text',
            'text': "You are a highly specialized AI image comparison assistant. "
        "Your ONLY task is to determine if two images contain the same person. "
        "Your response MUST be a single word: either 'Yes' or 'No'. "
        "Do not provide any explanation, punctuation, or any other text."}
    prompt = [task]
    for image in [person_image, image_to_label]:
        image_url = {'url':f'data:image/png;base64,{image}'}
        image_msg = {'type':'image_url', 'image_url':image_url}
        prompt += [image_msg]

    return prompt

def call_llm(ai_key, prompt):
    """ Call language model to process prompt with local images.

    Args:
        ai_key: key to access OpenAI.
        prompt: a prompt merging text and local images.

    Returns:
        answer by the language model, or an error message.
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {ai_key}'
    }
    payload = {
        'model': 'gpt-4o',
        'messages': [
            {'role': 'user', 'content': prompt}
            ],
        'max_tokens':1
        }
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers, json=payload)

    response_data = response.json()

    if 'choices' in response_data and response_data['choices']:
        return response_data['choices'][0]['message']['content']
    else:
        print("Error: API response does not contain 'choices'. Full response:")
        print(response_data)
        return "Error: Could not get response from API."

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

if __name__ == '__main__':
    """
  for Script Invoking from command line with argparse Ex- cd Tagging people in images  peopledir-- "/content/drive/MyDrive/Colab Notebooks/DataScience+GPT/Data/image and video/peoplepictures/people"
                                                                                       picsdir-- "/content/drive/MyDrive/Colab Notebooks/DataScience+GPT/Data/image and video/peoplepictures/pics"
                                                                                       outdir -- "/content/drive/MyDrive/Colab Notebooks/DataScience+GPT/Data/image and video/peoplepictures/processed"
  """

    parser = argparse.ArgumentParser()
    parser.add_argument('peopledir', type=str, help='Contains images of people')
    parser.add_argument('picsdir', type=str, help='Contains images to label')
    parser.add_argument('outdir', type=str, help='Contains processing result')
    args = parser.parse_args()

    people_images = load_images(args.peopledir)
    unlabeled_images = load_images(args.picsdir)

    for person_name, person_image in people_images.items():
        for un_name, un_image in unlabeled_images.items():
            prompt = create_prompt(person_image, un_image)
            ai_key = os.getenv('OPENAI_API_KEY')
            response = call_llm(ai_key, prompt)
            description = f'{un_name} versus {person_name}?'
            print(f'{description} -> {response}')

            if response == 'Yes':
                labeled_name = f'{person_name[:-4]}{un_name}'
                source_path = os.path.join(args.picsdir, un_name)
                target_path = os.path.join(args.outdir, labeled_name)
                shutil.copy(source_path, target_path)