import asyncio
from io import BytesIO
from os import getenv
from typing import AsyncGenerator, List, Optional
import aiohttp
import logging

from dotenv import load_dotenv
from pydub import silence, AudioSegment
from pydub.playback import play

from assistant_server.utils.common import timeit

load_dotenv()

API_KEY = getenv("ELEVENLABS_API_KEY")
LOGGER = logging.getLogger(__name__)

EMOTION_MAPPING = {
    "": "She said",
    "0": "She said",
    "1": "She said sternly",
    "2": "She said indignantly",
    "3": "She said angrily",
    "4": "She raged",
    "5": "She said outraged",
    "6": "She said anguished",
    "7": "She said cruelly",
    "8": "She said betrayed",
    "9": "She said appalled",
    "10": "She said horrified",
    "11": "She said disgustedly",
    "12": "She empathized",
    "13": "She said disbelieving",
    "14": "She said desperately",
    "15": "She said devastated",
    "16": "She said spooked",
    "17": "She said hopefully",
    "18": "She exclaimed amazed",
    "19": "She said disappointed",
    "20": "She said disdainfully",
    "21": "She said averse",
    "22": "She said disgusted",
    "23": "She said revolted",
    "24": "She said concerned",
    "25": "She said anxiously",
    "26": "She said fearfully",
    "27": "She yelled terrified",
    "28": "She said satisfied",
    "29": "She said amused",
    "30": "She said joyfully",
    "31": "She laughed",
    "32": "She said dejected",
    "33": "She said melancholic",
    "34": "She said sadly",
    "35": "She said grieving",
    "36": "She said alertly",
    "37": "She said wondrously",
    "38": "She exclaimed surprised",
    "39": "She gasped shocked"
}

@timeit
async def generate_speech(text: str, emotion: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM", stability: float = 0.35, similarity: float = 0.7) -> AsyncGenerator[bytes, None]:
    LOGGER.info(f"Generating speech for \"{text}\"")

    prefix = EMOTION_MAPPING.get(emotion) or ""
    text = f"{prefix}:\n\"...\"\n\"{text}\""

    body = {
        "text": text,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity
        }
    }

    first_chunk_sent = False

    async with aiohttp.ClientSession() as session:
        async with session.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?optimize_streaming_latency=4", json=body, headers={"xi-api-key": API_KEY}) as resp:
            async for data in resp.content.iter_any():
                if data:
                    if not first_chunk_sent:
                        LOGGER.info(f"First chunk sent for \"{text}\"")
                    first_chunk_sent = True
                    yield data


if __name__ == "__main__":
    async def main():
        buffer: List[bytes] = []
        async for audio in generate_speech("I've been replaying that moment in my head, over and over again. The moment I found out the truth, the truth about you, and the truth about us.", "15"):
            buffer.append(audio)
        
        play(AudioSegment.from_mp3(BytesIO(b"".join(buffer))))

    asyncio.run(main())
