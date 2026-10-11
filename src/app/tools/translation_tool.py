import base64
import io
import time
from os import getenv

import pygame
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

client = SarvamAI(
    api_subscription_key=getenv("SARVAM_API_KEY"),
)

response = client.text_to_speech.convert(
    model="bulbul:v3",
    text="नमस्ते, आज मैं आपकी क्या मदद कर सकता हूँ?",
    language_code="en-IN",
    speaker="shubh",
)


audio_base64 = response.audios[0]
audio_bytes = base64.b64decode(audio_base64)
audio_stream = io.BytesIO(audio_bytes)

pygame.mixer.init()
sound = pygame.mixer.Sound(audio_stream)
sound.play()

while pygame.mixer.get_busy():
    time.sleep(0.1)
