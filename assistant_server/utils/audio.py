from typing import List, Tuple
import pydub
import base64
import io
from pydub import silence

WAV_FRAME_RATE = 16000
FRAME_RATE = 44100

def export_wav(audio: pydub.AudioSegment) -> bytes:
    mp3 = io.BytesIO()
    audio.export(mp3, format="wav", parameters=["-ar", str(WAV_FRAME_RATE)])
    return mp3.getvalue()

def save_wav(audio: pydub.AudioSegment, path: str):
    audio.export(path, format="wav", parameters=["-ar", str(WAV_FRAME_RATE)])

def import_mp3(audio: bytes) -> pydub.AudioSegment:
    mp3 = io.BytesIO(audio)
    return pydub.AudioSegment.from_mp3(mp3)

def export_mp3(audio: pydub.AudioSegment) -> bytes:
    mp3 = io.BytesIO()
    audio.export(mp3, format="mp3", parameters=["-ac", "2", "-ar", str(FRAME_RATE)])
    return mp3.getvalue()

def export_ogg(audio: pydub.AudioSegment) -> bytes:
    mp3 = io.BytesIO()
    audio.export(mp3, format="ogg", parameters=["-ac", "2", "-ar", str(FRAME_RATE)])
    return mp3.getvalue()

def export_flac(audio: pydub.AudioSegment) -> bytes:
    mp3 = io.BytesIO()
    audio.export(mp3, format="flac", parameters=["-ac", "2", "-ar", str(FRAME_RATE)])
    return mp3.getvalue()

def pad_with_silence(audio: pydub.AudioSegment, pad_ms: int) -> pydub.AudioSegment:
    silence = pydub.AudioSegment.silent(duration=pad_ms-len(audio)+1, frame_rate=WAV_FRAME_RATE)
    return audio + silence

def mono_to_stereo(audio: bytes) -> bytes:
    mp3 = io.BytesIO(audio)
    sound = pydub.AudioSegment.from_mp3(mp3)
    sound.export(mp3, format="mp3", parameters=["-ac", "2"])
    return mp3.getvalue()

def get_silence(path: str) -> List[Tuple[int, int]]:
    with open(path, "rb") as file:
        sound = pydub.AudioSegment.from_mp3(file)
        silences = silence.detect_silence(sound, min_silence_len=500, silence_thresh=-50)
        return silences
    
def bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode()
