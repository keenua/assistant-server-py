import asyncio

from .gpt_server import start


if __name__ == "__main__":
    asyncio.run(start())
