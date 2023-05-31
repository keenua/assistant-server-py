from typing import List, Tuple
from pydub import AudioSegment

from assistant_server.server.utils import pad_with_silence

class AudioBuffer:
    def __init__(self, min_sound_length = 1000, silence_duration_ms: int = 2000, frame_rate=44100):
        self.silence = AudioSegment.silent(duration=silence_duration_ms, frame_rate=frame_rate)
        self.buffer: List[AudioSegment] = []
        self.min_sound_length = min_sound_length
        
    def append(self, audio: AudioSegment):
        self.buffer.append(audio)

    def sound_available(self) -> bool:
        return len(self.buffer) > 0
    
    def pop(self) -> Tuple[bool, AudioSegment]:
        if self.sound_available():
            sound = self.buffer.pop(0)
            sound = pad_with_silence(sound, self.min_sound_length)
            return True, sound

        return False, self.silence
