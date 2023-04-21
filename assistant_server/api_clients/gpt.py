import re
import os
import asyncio
from typing import Callable, List, Optional
import openai
from openai.openai_object import OpenAIObject
from dotenv import load_dotenv

load_dotenv()


class Statement:
    pass


class SayStatement(Statement):
    def __init__(self, text: str, emotion: str, gesture: str):
        self.text = text
        self.emotion = emotion
        self.gesture = gesture

    def __str__(self):
        return f"SayStatement(text=\"{self.text}\", emotion=\"{self.emotion}\", gesture=\"{self.gesture}\")"


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
        say_regex = r"<poop:(.*?):(.*?)>([\s\S]*?)<\/poop>"
        match = re.search(say_regex, self.buffer)

        if not match:
            return None

        emotion, gesture, text = match.groups()
        self.buffer = re.sub(say_regex, "", self.buffer)
        return SayStatement(text, emotion, gesture)

    def transform(self, chunk: OpenAIObject) -> Optional[list[Statement]]:
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


async def gpt(user_prompt: str, on_statement: Callable[[Statement], None]):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    with open(f"{current_dir}/configs/system_prompt.txt", "r") as f:
        system_prompt = f.read()

    print(f"Asking GPT-4 for {user_prompt}")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.7,
        stream=True
    )

    statement_transformer = StatementTransformer()
    for chunk in response:
        statements = statement_transformer.transform(chunk)

        if statements:
            for statement in statements:
                on_statement(statement)

    print(f"Full buffer: {statement_transformer.full_buffer}")


if __name__ == "__main__":
    def on_statement(statement: Statement):
        print(statement)

    user_prompt = "Write a Python function that takes two numbers and returns their sum."
    asyncio.run(gpt(user_prompt, on_statement))
