import asyncio
import os
from dataclasses import dataclass
from time import time
from typing import AsyncGenerator, List, Optional

from assistant_server.api_clients.speech import generate_speech
from assistant_server.gesture_generation.inference import GestureInferenceModel
from assistant_server.gesture_generation.visemes import Visemes
from assistant_server.server.utils import bytes_to_base64, mp3_to_wav


@dataclass
class StatementData:
    text: str
    emotion: str


@dataclass
class Frame:
    index: int
    motion: str
    audio: Optional[str]
    emotion: Optional[str]
    text: Optional[str]
    viseme: Optional[str]


class Director:
    def __init__(self, preferred_buffer_time: float = 2.0) -> None:
        self.FPS = 60
        self.PREFFERED_BUFFER_TIME = preferred_buffer_time

        self.frame_index: int = 0
        self.statement_queue: List[StatementData] = []

        self.gesture_model = GestureInferenceModel()
        self.gesture_model.load_model()

        self.viseme_model = Visemes()

        self.start_time = time()

        self.frame_buffer: List[Frame] = []
        self.processing = False

    def add_statement(self, statement: StatementData) -> None:
        print(f"Statement added: {statement.text}")
        self.processing = True
        self.statement_queue.append(statement)

    def is_idle(self):
        return not self.processing

    def get_buffer_time(self) -> float:
        delta = time() - self.start_time
        return self.frame_index / self.FPS - delta

    def buffered(self) -> bool:
        return self.get_buffer_time() >= self.PREFFERED_BUFFER_TIME

    async def __fill_audio_buffer(self, text: str, buffer: List[bytes]) -> None:
        print(f"Generating audio for: {text}")
        async for audio in generate_speech(text):
            buffer.append(audio)

    def __generate_from_audio(self, audio: Optional[bytes], emotion: Optional[str], text: Optional[str]) -> List[Frame]:
        wav_file = "temp.wav" if audio else None
        audio_base64: Optional[str] = None

        if wav_file and audio:
            mp3_to_wav(audio, wav_file)
            audio_base64 = bytes_to_base64(audio)

        motions = self.gesture_model.infer_motions(wav_file)
        visemes = self.viseme_model.recognize(wav_file) if audio else []

        if wav_file:
            os.remove(wav_file)

        result: List[Frame] = []

        for motion in motions:
            timestamp = self.frame_index / float(self.FPS)

            viseme = visemes.pop(0).viseme if visemes and timestamp >= visemes[0].start else None

            frame = Frame(
                index=self.frame_index,
                motion=motion.strip(),
                audio=audio_base64,
                emotion=emotion,
                text=text,
                viseme=viseme,
            )
            result.append(frame)

            self.frame_index += 1

            # Only send audio and text on the first frame
            audio_base64 = None
            text = None
            emotion = None

        return result

    def __generate_silence(self) -> List[Frame]:
        return self.__generate_from_audio(None, None, "")

    async def __generate_frames(self) -> AsyncGenerator[Frame, None]:
        buffer: List[bytes] = []
        fill_buffer_task: Optional[asyncio.Task] = None

        while True:
            frames: List[Frame] = []
            task_empty = fill_buffer_task is None or fill_buffer_task.done()

            if self.statement_queue and task_empty:
                statement = self.statement_queue.pop(0)
                fill_buffer_task = asyncio.create_task(self.__fill_audio_buffer(statement.text, buffer))

            if buffer:
                audio = buffer.pop(0)
                frames = self.__generate_from_audio(audio, statement.emotion, statement.text)
            elif not self.buffered():
                frames = self.__generate_silence()

            for frame in frames:
                yield frame

            self.processing = not task_empty or bool(buffer)

            print(f"Buffer: {self.get_buffer_time()}")
            await asyncio.sleep(0.1)

    async def __run(self):
        async for frame in self.__generate_frames():
            self.frame_buffer.append(frame)

    def start(self):
        self.start_time = time()
        self.statement_queue.clear()
        self.frame_index = 0
        self.frame_buffer.clear()
        asyncio.ensure_future(self.__run())

    def get_frames(self, max_frames: int = 0) -> List[Frame]:
        frames_to_take = max_frames if max_frames else self.FPS
        frames = self.frame_buffer[:frames_to_take]
        self.frame_buffer = self.frame_buffer[frames_to_take:]
        return frames
