import asyncio
from io import BytesIO
from os import getenv
from typing import AsyncGenerator, List, Optional
import aiohttp

from dotenv import load_dotenv
from elevenlabs import play
from pydub import silence, AudioSegment

load_dotenv()

API_KEY = getenv("ELEVENLABS_API_KEY")

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

def split_by_silence(audio: bytes) -> Optional[bytes]:
    mp3 = BytesIO(audio)
    sound = AudioSegment.from_mp3(mp3)
    silences = silence.detect_silence(sound, min_silence_len=300, silence_thresh=-50)

    if (len(silences) == 0):
        return None

    start, end = silences[0]
    mid = start + (end - start) * 3 // 4 

    mp3 = BytesIO()
    sound[mid:].export(mp3, format="mp3")
    return mp3.getvalue()

async def generate_speech_with_prefix(text: str, emotion: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM", stability: float = 0.35, similarity: float = 0.7) -> AsyncGenerator[bytes, None]:
    prefix = EMOTION_MAPPING.get(emotion) or ""
    text = f"{prefix}:\n\"...\"\n\"{text}\""

    body = {
        "text": text,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?optimize_streaming_latency=3", json=body, headers={"xi-api-key": API_KEY}) as resp:
            async for data in resp.content.iter_any():
                if data:
                    yield data

async def generate_speech(text: str, emotion: str, chunk_size: int = 8000 * 2, voice_id: str = "21m00Tcm4TlvDq8ikWAM", stability: float = 0.35, similarity: float = 0.7) -> AsyncGenerator[bytes, None]:
    buffer: List[bytes] = []

    async for audio in generate_speech_with_prefix(text, emotion, voice_id, stability, similarity):
        buffer.append(audio)
    audio = b"".join(buffer)
    result = split_by_silence(audio)
    if result:
        yield result
    else:
        yield audio


if __name__ == "__main__":
    async def main():
        buffer = BytesIO()
        async for audio in generate_speech("I've been replaying that moment in my head, over and over again. The moment I found out the truth, the truth about you, and the truth about us.", "15"):
            buffer.write(audio)
        
        buffer.seek(0)
        play(buffer.getvalue())

    asyncio.run(main())
