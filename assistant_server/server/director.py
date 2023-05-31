import asyncio
from io import BytesIO
import json
import os

from dataclasses import dataclass
from time import time
from uuid import uuid4
from typing import AsyncGenerator, List, Optional
from pydub import AudioSegment

from assistant_server.api_clients.speech import generate_speech
from assistant_server.gesture_generation.inference import GestureInferenceModel
from assistant_server.gesture_generation.visemes import Visemes
from assistant_server.server.audio import AudioBuffer
from assistant_server.server.utils import bytes_to_base64, export_flac, save_wav


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

EMOTION_TO_STYLE = {
  "1": "Neutral",
  "2": "Angry",
  "3": "Angry",
  "4": "Angry",
  "5": "Angry",
  "6": "Scared",
  "7": "Angry",
  "8": "Sad",
  "9": "Pensive",
  "10": "Scared",
  "11": "Happy",
  "12": "Sad",
  "13": "Pensive",
  "14": "Scared",
  "15": "Sad",
  "16": "Scared",
  "17": "Sad",
  "18": "Happy",
  "19": "Sad",
  "20": "Disagreement",
  "21": "Disagreement",
  "22": "Disagreement",
  "23": "Disagreement",
  "24": "Neutral",
  "25": "Scared",
  "26": "Scared",
  "27": "Scared",
  "28": "Happy",
  "29": "Laughing",
  "30": "Happy",
  "31": "Laughing",
  "32": "Sad",
  "33": "Sad",
  "34": "Sad",
  "35": "Sad",
  "36": "Neutral",
  "37": "Happy",
  "38": "Pensive",
  "39": "Pensive"
}

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
        self.sound_duration_ms = 0
        self.frames_played = 0

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

    async def __fill_audio_buffer(self, text: str, emotion: str, buffer: AudioBuffer) -> None:
        print(f"Generating audio for: {text}")
        async for audio in generate_speech(text, emotion):
            await buffer.append(audio)

        await buffer.flush()

    def __generate_from_audio(self, available: bool, audio: AudioSegment, emotion: Optional[str], text: Optional[str]) -> List[Frame]:
        start_frame_index = self.frame_index
        wav_file = f"{uuid4()}.wav"
        audio_base64: Optional[str] = None

        save_wav(audio, wav_file)
        
        visemes = self.viseme_model.recognize(wav_file) if available else []
        
        style = EMOTION_TO_STYLE[emotion] if emotion else "Neutral"
        motions = self.gesture_model.infer_motions(style, wav_file)

        frames_shift = int(self.sound_duration_ms / 1000 * self.FPS) - self.frames_played

        if frames_shift > 0:
            # repeat first frame
            for _ in range(frames_shift):
                motions.insert(0, motions[0])
        elif frames_shift < 0:
            # remove first frames
            motions = motions[-frames_shift:]

        if wav_file:
            os.remove(wav_file)

        result: List[Frame] = []

        viseme_end = 0
        last_viseme_was_silence = True
        offset = 0.1

        first_frame = True

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

            if first_frame:
                sound_bytes = export_flac(audio)
                audio_base64 = bytes_to_base64(sound_bytes)
                self.sound_duration_ms += len(audio)
            else:
                audio_base64 = None

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
            first_frame = False
            audio_base64 = None
            text = None
            emotion = None

            self.frames_played += 1

        return result

    async def __generate_frames(self) -> AsyncGenerator[Frame, None]:
        buffer = AudioBuffer()
        fill_buffer_task: Optional[asyncio.Task] = None
        statement: Optional[StatementData] = None

        while True:
            frames: List[Frame] = []
            task_empty = fill_buffer_task is None or fill_buffer_task.done()

            if self.statement_queue and task_empty:
                statement = self.statement_queue.pop(0)
                fill_buffer_task = asyncio.create_task(self.__fill_audio_buffer(statement.text, statement.emotion, buffer))

            if (await buffer.sound_available() and statement) or not self.buffered():
                available, audio = await buffer.pop()
                emotion = statement.emotion if statement else None
                text = statement.text if statement else None
                frames = self.__generate_from_audio(available, audio, emotion, text)

            for frame in frames:
                yield frame

            self.processing = not task_empty or await buffer.sound_available()

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
