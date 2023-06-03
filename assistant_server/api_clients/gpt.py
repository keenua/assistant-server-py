import asyncio
import json
import logging
import os
import re
from typing import AsyncGenerator, Iterator, List, Optional

import aiohttp
from dotenv import load_dotenv

from assistant_server.utils.common import timeit

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

LOGGER = logging.getLogger(__name__)


class Statement:
    pass


class SayStatement(Statement):
    def __init__(self, text: str, emotion: str):
        self.text = text
        self.emotion = emotion

    def __str__(self):
        return f"SayStatement(text=\"{self.text}\", emotion=\"{self.emotion}\")"


class CodeStatement(Statement):
    def __init__(self, code: str, language: str):
        self.code = code
        self.language = language

    def __str__(self):
        return f"CodeStatement(code=\"{self.code}\", language=\"{self.language}\")"


class StatementTransformer:
    def __init__(self):
        self.buffer = ""
        self.full_buffer = ""

    def parse_code_from_buffer(self) -> Optional[CodeStatement]:
        code_regex = r"```(.*?)\n([\s\S]*?)```"
        match = re.search(code_regex, self.buffer)

        if not match:
            return None

        language, code = match.groups()
        self.buffer = re.sub(code_regex, "", self.buffer)
        return CodeStatement(code.strip(), language)

    def parse_say_from_buffer(self) -> Optional[SayStatement]:
        say_regex = r"<poop:(.*?)>([\s\S]*?)<\/poop>"
        match = re.search(say_regex, self.buffer)

        if not match:
            return None

        emotion, text = match.groups()
        self.buffer = re.sub(say_regex, "", self.buffer)
        return SayStatement(text, emotion)

    def transform(self, chunk: dict) -> Optional[list[Statement]]:
        if not chunk.get("choices"):
            return None

        delta = chunk["choices"][0].get("delta")

        if not delta:
            return None

        content = delta.get("content")

        if not content:
            return None

        self.buffer += content
        self.full_buffer += content

        statements: List[Statement] = []
        statement = None
        while statement := self.parse_code_from_buffer() or self.parse_say_from_buffer():
            statements.append(statement)

        return statements


def parse_stream_helper(line: bytes) -> Optional[str]:
    if line:
        if line.strip() == b"data: [DONE]":
            # return here will cause GeneratorExit exception in urllib3
            # and it will close http connection with TCP Reset
            return None
        if line.startswith(b"data: "):
            line = line[len(b"data: "):]
            return line.decode("utf-8")
        else:
            return None
    return None


def parse_stream(rbody: Iterator[bytes]) -> Iterator[str]:
    for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line


async def parse_stream_async(rbody: aiohttp.StreamReader):
    async for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line


async def stream(prompt: str, system_prompt: str, model: str) -> AsyncGenerator[str, None]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,
        "n": 1,
        "temperature": 0.7,
        "stream": True
    }

    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.openai.com/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {API_KEY}"}) as resp:
            async for data in parse_stream_async(resp.content):
                yield data


async def gpt(user_prompt: str) -> AsyncGenerator[Statement, None]:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    with open(f"{current_dir}/configs/system_prompt.txt", "r") as f:
        system_prompt = f.read()

    LOGGER.info(f"Asking GPT-4 for {user_prompt}")
    statement_transformer = StatementTransformer()

    async for data in stream(user_prompt, system_prompt, "gpt-4"):
        chunk = json.loads(data)

        statements = statement_transformer.transform(chunk)

        if statements:
            for statement in statements:
                yield statement

    LOGGER.info(f"Full buffer: {statement_transformer.full_buffer}")


async def test():
    user_prompt = "Write a Python function that takes two numbers and returns their sum."
    async for statement in gpt(user_prompt):
        LOGGER.info(statement)


if __name__ == "__main__":
    asyncio.run(test())
