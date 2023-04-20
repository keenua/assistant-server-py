from os import getenv

from dotenv import load_dotenv
from elevenlabs import generate, play, set_api_key

load_dotenv()

API_KEY = getenv("ELEVENLABS_API_KEY")
set_api_key(API_KEY)


def generate_speech(text: str) -> bytes:
    voice = "Rachel"
    audio = generate(text=text, voice=voice)
    return audio


if __name__ == "__main__":
    audio = generate_speech("Hello world!")
    play(audio)
