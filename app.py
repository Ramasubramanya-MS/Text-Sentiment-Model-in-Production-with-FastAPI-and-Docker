## ML Model Deployment at Streamlit Server
# Full Streamlit Code Repository: https://github.com/laxmimerit/streamlit-tutorials

# streamlit run 2-app.py

import streamlit as st
import os
import torch
from transformers import pipeline
import requests
from PIL import Image
from io import BytesIO
import random
import boto3
import json
bucket_name = "mlops-29-12-24"
s3 = boto3.client('s3')
local_path = 'tinybert-sentiment-analysis'
s3_prefix = 'ml-models/tinybert-sentiment-analysis/'

apikey = "AIzaSyCb0l_aqa49taTTtJURcEWWM5j9xErkxI8"  # click to set to your apikey
lmt = 8
ckey = "my_test_app"  # set the client_key for the integration and use the same value for all API calls

def fetch_gif(search_term):
    """Fetch a random GIF URL from Tenor API based on the search term."""
    url = f"https://tenor.googleapis.com/v2/search?q={search_term}&key={apikey}&client_key={ckey}&limit={lmt}"
    response = requests.get(url)
    if response.status_code == 200:
        gifs = json.loads(response.content)
        if gifs.get("results"):
            # Return a random GIF URL from the results
            return random.choice(gifs["results"])["media_formats"]["gif"]["url"]
    return None


def download_dir(local_path, s3_prefix):
    # Ensure the local directory exists
    os.makedirs(local_path, exist_ok=True)

    # Use a paginator to handle large folders
    paginator = s3.get_paginator('list_objects_v2')

    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for obj in result['Contents']:
                s3_key = obj['Key']

                # Compute the local file path
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))

                # Ensure the directory for the local file exists
                os.makedirs(os.path.dirname(local_file), exist_ok=True)

                # Download the file from S3
                print(f"Downloading {s3_key} to {local_file}")
                s3.download_file(bucket_name, s3_key, local_file)


st.title("TextVibe Analysis AI App")

button = st.button("Download Model")
if button:
    with st.spinner("Downloading... Please wait!"):
        download_dir(local_path, s3_prefix)


text = st.text_area("Enter Your Review", "Type...")
predict = st.button("Predict")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
classifier = pipeline('text-classification', model='tinybert-sentiment-analysis', device=device)
if predict:
    
    with st.spinner("Predicting..."):
        output = classifier(text)
        # st.json(output)
        label = output[0]['label']
        score = output[0]['score']
        # Fetch a GIF based on sentiment
        if label == "positive":
            gif_url = fetch_gif("happy positive anime")
            if gif_url:
                st.subheader(f"Label: {label}")
                st.subheader(f"Score: {score}")
                st.image(gif_url, caption="Positive vibes in a GIF!")
        elif label == "negative":
            gif_url = fetch_gif("sad anime")
            if gif_url:
                st.subheader(f"Label: {label}")
                st.subheader(f"Score: {score}")
                st.image(gif_url, caption="Negative vibes in a GIF!")