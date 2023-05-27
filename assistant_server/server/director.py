import asyncio
import json
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

        self.run_task: Optional[asyncio.Task] = None

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
            print("Adding chunk to buffer (%d)" % len(buffer))

    def __generate_from_audio(self, audio: Optional[bytes], emotion: Optional[str], text: Optional[str]) -> List[Frame]:
        start_frame_index = self.frame_index
        wav_file = "temp.wav" if audio else None
        audio_base64: Optional[str] = None

        if wav_file and audio:
            mp3_to_wav(audio, wav_file)
            audio_base64 = bytes_to_base64(audio)

        motions = self.gesture_model.infer_motions(wav_file)
        visemes = self.viseme_model.recognize(wav_file) if wav_file else []

        if wav_file:
            os.remove(wav_file)

        result: List[Frame] = []

        viseme_end = 0
        last_viseme_was_silence = True
        offset = 0.1

        for motion in motions:
            timestamp = (self.frame_index - start_frame_index) / float(self.FPS)

            viseme = visemes.pop(0) if visemes and timestamp >= visemes[0].start - offset else None
            
            if viseme:
                viseme_end = viseme.start + viseme.duration
                last_viseme_was_silence = False
            
            viseme_str = viseme.viseme if viseme else ""

            if not viseme and timestamp >= viseme_end and not last_viseme_was_silence:
                viseme_str = "sil"
                last_viseme_was_silence = True

            frame = Frame(
                index=self.frame_index,
                motion=motion.strip(),
                audio=audio_base64 or "",
                emotion=emotion or "",
                text=text or "",
                viseme=viseme_str
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
        statement: Optional[StatementData] = None

        while True:
            frames: List[Frame] = []
            task_empty = fill_buffer_task is None or fill_buffer_task.done()

            if self.statement_queue and task_empty:
                statement = self.statement_queue.pop(0)
                fill_buffer_task = asyncio.create_task(self.__fill_audio_buffer(statement.text, buffer))

            if buffer and statement:
                print("Popping chunk from buffer (%d)" % len(buffer))
                audio = buffer.pop(0)
                print("Generating from audio (%d)" % len(audio))
                frames = self.__generate_from_audio(audio, statement.emotion, statement.text)
                print("Generated from audio (%d)" % len(audio))
            elif not self.buffered():
                frames = self.__generate_silence()
                # frames = []

            for frame in frames:
                yield frame

            self.processing = not task_empty or bool(buffer)

            # print(f"Buffer: {self.get_buffer_time()}")
            await asyncio.sleep(0.1)

    async def __generate_frames_from_files(self, dir: str) -> AsyncGenerator[Frame, None]:
        files = sorted(os.listdir(dir), key=lambda x: int(x[4:-5]))

        for file in files:
            with open(os.path.join(dir, file), "r") as f:
                data = json.load(f)

            for frame in data:
                yield Frame(**frame)
            
    async def __run(self, dir: Optional[str] = None):
        generator = self.__generate_frames_from_files(dir) if dir else self.__generate_frames()
        async for frame in generator:
            self.frame_buffer.append(frame)

    def start(self, frames_dir: Optional[str] = None):
        self.start_time = time()
        self.statement_queue.clear()
        self.frame_index = 0
        self.frame_buffer.clear()

        if self.run_task:
            self.run_task.cancel()

        self.run_task = asyncio.ensure_future(self.__run(frames_dir))

    def stop(self):
        if self.run_task:
            self.run_task.cancel()

    def get_frames(self, max_frames: int = 0) -> List[Frame]:
        frames_to_take = max_frames if max_frames else self.FPS
        frames = self.frame_buffer[:frames_to_take]
        self.frame_buffer = self.frame_buffer[frames_to_take:]
        return frames
