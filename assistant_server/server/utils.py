import os
from typing import List, Tuple
import pydub
import base64
import io
from elevenlabs import play
from pydub import silence

FRAME_RATE = 16000

def mp3_to_wav(audio: bytes, dest_path: str, pad_ms = 1000) -> pydub.AudioSegment:
    mp3 = io.BytesIO(audio)
    sound: pydub.AudioSegment = pydub.AudioSegment.from_mp3(mp3)

    sound = pad_with_silence(sound, pad_ms)
    sound.export(dest_path, format="wav", parameters=["-ar", "16000"])
    return sound

def export_mp3(audio: pydub.AudioSegment) -> bytes:
    mp3 = io.BytesIO()
    audio.export(mp3, format="mp3", parameters=["-ac", "2", "-ar", "44100"])
    return mp3.getvalue()

def export_ogg(audio: pydub.AudioSegment) -> bytes:
    mp3 = io.BytesIO()
    audio.export(mp3, format="ogg", parameters=["-ac", "2", "-ar", "44100"])
    return mp3.getvalue()

def pad_with_silence(audio: pydub.AudioSegment, pad_ms: int) -> pydub.AudioSegment:
    silence = pydub.AudioSegment.silent(duration=pad_ms-len(audio)+1, frame_rate=FRAME_RATE)
    return audio + silence

def correct_length(audio: pydub.AudioSegment, ms: int) -> pydub.AudioSegment:
    if len(audio) < ms:
        return pad_with_silence(audio, ms)
    else:
        return audio[:ms]

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


def merge_frames(base_file: str, from_folder: str):
    import os
    import json

    with open(base_file, "r") as f:
        headers = f.readlines()[:464]

    frames = []
    for file in sorted(os.listdir(from_folder), key=lambda x: int(x[4:-5])):
        print(file)
        with open(f"{from_folder}/{file}", "r") as f:
            frames += json.load(f)

    with open("merged.bvh", "w") as f:
        f.writelines(headers)

        for frame in frames:
            f.write(frame["motion"] + "\n")

def visualize_logs():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dir = f"{current_dir}/../../data/samples/frames/football"

    import json

    # for each file in dir
    #   for each frame in file
    for file in sorted(os.listdir(dir), key=lambda x: int(x[4:-5])):
        with open(f"{dir}/{file}", "r") as f:
            frames = json.load(f)

            for frame in frames:
                if frame["text"] != "" or frame["audio"] != "":
                    print(f)
                    print(frame["index"])
                    print(frame["text"])
                    print(len(frame["audio"]))
                    audio_bytes = base64.b64decode(frame["audio"])
                    play(audio_bytes)
                    print("-----")
                    input()

def print_all_emotions(dir: str):
    file_names = []
    for file in sorted(os.listdir(dir)):
        if file.endswith(".uasset"):
            file_names.append(file[:-7])

    print(file_names)

if __name__ == "__main__":
    # merge_frames("data/zeggs/styles/relaxed_fixed.bvh", "data/samples/frames/divorce_offset_3")
    # visualize_logs()
    # print_all_emotions("e:\\Work\\Projects\\test_mh\\Assistant\\Content\\Animations\\Jawless")


    # for each dir in dir
    for dir in os.listdir("data/samples/sounds"):
        for file in os.listdir(f"data/samples/sounds/{dir}"):
            if file.endswith(".mp3"):
                file_path = f"data/samples/sounds/{dir}/{file}"
                print(file_path)
                silences = get_silence(file_path)
                input(silences)
