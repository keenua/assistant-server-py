from typing import Optional, Tuple
from pydub import AudioSegment, silence

from assistant_server.server.utils import import_mp3, pad_with_silence
from asyncio import Lock

class AudioBuffer:
    def __init__(self, min_sound_length_ms = 1000, chunk_length_ms = 2000, silence_duration_ms: int = 2000, frame_rate=44100):
        self.silence = AudioSegment.silent(duration=silence_duration_ms, frame_rate=frame_rate)
        self.buffer: bytes = b""
        self.min_sound_length = min_sound_length_ms
        self.chunk_length_ms = chunk_length_ms
        self.sound: AudioSegment = AudioSegment.empty()
        self.leftover_sound: Optional[AudioSegment] = None
        self.read = 0
        self.lock = Lock()

        
    async def append(self, audio: bytes):
        async with self.lock:
            print(f"Appending {len(audio)} bytes to buffer")
            self.buffer += audio
            self.sound = import_mp3(self.buffer)

            if self.read == 0:
                self.read = self.detect_silence()

    async def flush(self):
        async with self.lock:
            print(f"Flushing buffer of size {len(self.buffer)}")
            self.leftover_sound = import_mp3(self.buffer)[self.read:] # type: ignore
            self.leftover_sound = pad_with_silence(self.leftover_sound, self.min_sound_length) # type: ignore
            print(f"Leftover sound is {len(self.leftover_sound)} ms")
            self.sound = AudioSegment.empty()
            self.buffer = b""
            self.read = 0

    async def sound_available(self) -> bool:
        async with self.lock:
            return (len(self.sound) > self.read + self.min_sound_length and self.read > 0) or self.leftover_sound is not None
    
    async def pop(self) -> Tuple[bool, AudioSegment]:
        async with self.lock:
            if self.leftover_sound:
                print(f"Returning leftover sound of size {len(self.leftover_sound)} ms")
                sound = self.leftover_sound
                self.leftover_sound = None
                return True, sound

            if len(self.sound) > self.read + self.chunk_length_ms:
                print(f"Returning sound from {self.read} to {self.read+self.chunk_length_ms} ms, total size {len(self.sound)} ms")
                sound: AudioSegment = self.sound[self.read:self.read+self.chunk_length_ms] # type: ignore
                self.read += self.chunk_length_ms
                return True, sound

            print(f"Returning silence of size {len(self.silence)} ms")
            return False, self.silence
        
    def detect_silence(self) -> int:
        silences = silence.detect_silence(self.sound, min_silence_len=300, silence_thresh=-50)

        if (len(silences) == 0):
            return 0

        start, end = silences[0]
        mid = start + (end - start) * 3 // 4 

        return mid