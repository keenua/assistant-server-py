import pydub
import base64
import io


def mp3_to_wav(audio: bytes, dest_path: str) -> bytes:
    mp3 = io.BytesIO(audio)
    sound = pydub.AudioSegment.from_mp3(mp3)
    sound = sound.set_frame_rate(16000)
    sound.export(dest_path, format="wav")

    with open(dest_path, "rb") as file:
        return file.read()


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


if __name__ == "__main__":
    merge_frames("data/results/sample_motion.bvh", "data/results/frames")
