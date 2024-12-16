import itertools
import json
import os
import sys
import threading
import time

import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from random_user_agent.params import OperatingSystem, SoftwareName
from random_user_agent.user_agent import UserAgent
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


SEARCH_TERM = "xrp"
MAX_VIDEOS = 3
# last 24hrs, Video only, 4-20 min, sorted by view count
FILTER = "CAMSBggCEAEYAw%253D%253D"

software_names = [SoftwareName.CHROME.value]
operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]

user_agent_rotator = UserAgent(
    software_names=software_names, operating_systems=operating_systems, limit=100
)

# Get list of user agents.
user_agents = user_agent_rotator.get_user_agents()

# Get Random User Agent String.
user_agent = user_agent_rotator.get_random_user_agent()


genai.configure(api_key=api_key)
# Create the model
generation_config = {
    "temperature": 0.5,
    "top_p": 0.8,
    "top_k": 30,
    "max_output_tokens": 8192,  # how much output the AI can produce
    "response_mime_type": "text/plain",
}


model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

chat_session = model.start_chat(history=[])

first_prompt = f"""

You area a financial analyst assistant. Your task is to analyze video transcripts discussing XRP (Ripple) and provide a summary of the key points, focusing on market trends, recent developments, and predictions. 
I will be giving you a total of around {MAX_VIDEOS} video transcripts 


Please remember this information while I you analyze these videos, I will be asking you to answer me questions after.
"""

last_prompt = """
Okay after analyzing these videos, give me a short summary of all video combined into 3 sentences giving me detaisl of what know about XRP. 

Secondly I want you to tell me how many out of the videos what each move the video said to do, Buy, Sell, or Hold. List the video number then - move

Thirdly I want you to tell me the suggested move I should do in one word, either Buy, Sell, or Hold. 


"""


def get_yt_ids():
    print("Getting Youtube Video IDs...")

    headers = {"User-Agent": user_agent}

    request = requests.get(
        f"https://www.youtube.com/results?search_query={SEARCH_TERM}&sp={FILTER}",
        headers=headers,
    )
    # print(request.text)
    soup = BeautifulSoup(request.text, "lxml")

    # gets the script where it has the video data inside (it is a JSON format so we turn it into JSON object in python)
    script_tag = soup.find(
        "script", string=lambda text: text and "ytInitialData" in text
    )
    script_data = script_tag.string[20:][:-1]
    script_data_in_json = json.loads(script_data)
    youtube_video_ids = []
    for i in range(MAX_VIDEOS):
        video_id = (
            script_data_in_json.get("contents")
            .get("twoColumnSearchResultsRenderer")
            .get("primaryContents")
            .get("sectionListRenderer")
            .get("contents")[0]
            .get("itemSectionRenderer")
            .get("contents")[i]  # this is where we find the index i video data
            .get("videoRenderer")
            .get("videoId")  # video ID
        )
        youtube_video_ids.append(video_id)

    print("Done!")

    return youtube_video_ids


def get_yt_transcripts(video_ids):
    print("Getting Youtube Video Transcripts...")

    video_transcripts = {}
    for i in range(len(video_ids)):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_ids[i])
        except NoTranscriptFound as e:
            print("English transcript not found, skipping video...")
        full_transcript = ""
        for entry in transcript_list:
            full_transcript += entry.get("text")
            full_transcript += " "

        video_transcripts[video_ids[i]] = full_transcript

    print("Done!")

    return video_transcripts


def analyze_transcripts(transcripts):
    print("Analyzing transcripts...")

    chat_session.send_message(first_prompt)  # gives it context

    for key, value in transcripts.items():
        main_prompt = f"""
        Here is a transcript of a video about XRP recorded in the last 24 hours
        Again follow what I said and remember this video context in order to understand what my next move should be:
        {value}
        """
        response = chat_session.send_message(main_prompt)
        # print(response)

    conclusion = chat_session.send_message(last_prompt)
    print("Done!\n")
    print(conclusion.text)


if __name__ == "__main__":
    video_ids = get_yt_ids()
    video_transcripts = get_yt_transcripts(video_ids)
    analyze_transcripts(video_transcripts)
