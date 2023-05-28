import os
import pydub
import base64
import io
from elevenlabs import play

def mp3_to_wav(audio: bytes, dest_path: str) -> bytes:
    mp3 = io.BytesIO(audio)
    sound = pydub.AudioSegment.from_mp3(mp3)
    sound = sound.set_frame_rate(16000)
    sound.export(dest_path, format="wav")

    with open(dest_path, "rb") as file:
        return file.read()

def mono_to_stereo(audio: bytes) -> bytes:
    mp3 = io.BytesIO(audio)
    sound = pydub.AudioSegment.from_mp3(mp3)
    sound.export(mp3, format="mp3", parameters=["-ac", "2"])
    return mp3.getvalue()

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
    merge_frames("data/zeggs/styles/relaxed_fixed.bvh", "data/samples/frames/divorce_offset_3")
    # visualize_logs()
    # print_all_emotions("e:\\Work\\Projects\\test_mh\\Assistant\\Content\\Animations\\Jawless")
