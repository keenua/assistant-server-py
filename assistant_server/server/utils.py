import pydub
import base64
import io


def mp3_to_wav(audio: bytes, dest_path: str) -> bytes:
    mp3 = io.BytesIO(audio)
    sound = pydub.AudioSegment.from_mp3(mp3)
    sound.export(dest_path, format="wav")

    with open(dest_path, "rb") as file:
        return file.read()


def bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode()
