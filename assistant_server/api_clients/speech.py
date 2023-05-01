import asyncio
from os import getenv
from typing import AsyncGenerator
import aiohttp

from dotenv import load_dotenv
from elevenlabs import play

load_dotenv()

API_KEY = getenv("ELEVENLABS_API_KEY")


async def stream(text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM", stability: float = 0.35, similarity: float = 0.7) -> AsyncGenerator[bytes, None]:
    body = {
        "text": text,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream", json=body, headers={"xi-api-key": API_KEY}) as resp:
            async for data, _ in resp.content.iter_chunks():
                yield data


async def get_chunks(generator: AsyncGenerator[bytes, None], chunk_size: int) -> AsyncGenerator[bytes, None]:
    chunk = b""
    async for data in generator:
        chunk += data
        if len(chunk) >= chunk_size * 3 / 2:
            yield chunk[:chunk_size]
            chunk = chunk[chunk_size:]
    if chunk:
        yield chunk


async def generate_speech(
    text: str,
    chunk_size: int = 8000 * 2,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    stability: float = 0.35,
    similarity: float = 0.7
) -> AsyncGenerator[bytes, None]:
    async for chunk in get_chunks(stream(text, voice_id, stability, similarity), chunk_size):
        yield chunk


if __name__ == "__main__":
    async def main():
        async for audio in generate_speech("I've been replaying that moment in my head, over and over again. The moment I found out the truth, the truth about you, and the truth about us. You know, I've always had this vision of the perfect relationship, and I thought I had found it in you. I thought we were a team, two people who had each other's back, no matter what. But I guess I was wrong."):
            play(audio)

    asyncio.run(main())
