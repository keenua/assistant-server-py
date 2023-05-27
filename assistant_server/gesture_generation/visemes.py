from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Tuple

from allosaurus.app import read_recognizer
from allosaurus.lm.inventory import Inventory
from allosaurus.model import get_model_path
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from assistant_server.gesture_generation.utils import timeit

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

@dataclass
class Frame:
    start: float
    duration: float
    phone: str
    viseme: str


class VisemeMovieMaker():
    VISEME_TO_FILE = {
        "": "sil",
        "k": "k",
        "p": "p",
        "r": "r",
        "f": "f",
        "@": "@",
        "a": "a",
        "e": "e",
        "i": "i",
        "o": "o",
        "O": "0",
        "u": "u",
        "E": "EE",
        "t": "t",
        "T": "TT",
        "s": "s",
        "S": "SS"
    }

    def __init__(self, out_dir: str, fps=60) -> None:
        self.out_dir = out_dir

        self.im_dir = f"{CURRENT_DIR}/../../data/visemes/images"
        self.height, self.width = self.get_im_dims(self.im_dir)
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.fps = fps
        self.generator = Visemes()

    def get_im_dims(self, im_dir: str) -> Tuple[int, int]:
        for image in os.listdir(im_dir):
            try:
                frame = cv2.imread(os.path.join(im_dir, image))
                height, width, _ = frame.shape
                return height, width
            except:
                continue

        raise Exception("No images found in directory")

    def get_out(self, out_path: str) -> cv2.VideoWriter:
        return cv2.VideoWriter(out_path, self.fourcc, self.fps, (self.width, self.height))

    def make_frame(self, viseme: str) -> np.ndarray:
        # allow other file types
        frame = cv2.imread(os.path.join(self.im_dir, f"{viseme}.jpeg"))
        return cv2.resize(frame, (self.width, self.height))

    def generate_video(self, in_file: str):
        file_name = os.path.basename(in_file).split(".")[0]
        
        audio = AudioFileClip(in_file)
        audio_dur = audio.duration
        audio.close()

        self.out_path = (os.path.join(self.out_dir, f'{file_name}.mp4'))
        
        output = self.get_out(self.out_path)
        phones = self.generator.recognize(in_file)
        
        for frame_index in range(0, int(audio_dur * self.fps)):
            current_timestamp = frame_index / self.fps
            while phones and phones[0].start + phones[0].duration <= current_timestamp:
                phones.pop(0)

            current_phone = phones[0] if phones else None
            
            if current_phone and current_phone.start > current_timestamp:
                current_phone = None

            mapped = self.VISEME_TO_FILE[current_phone.viseme if current_phone else ""]
            cv_frame = self.make_frame(mapped)
            output.write(cv_frame)
    
        output.release()
        cv2.destroyAllWindows()
        print(
            f"Generated video of {audio_dur} milliseconds from viseme images.")

    def add_audio(self, audio_file: str, video_file: str):
        video_clip = VideoFileClip(video_file)
        audio_clip = AudioFileClip(audio_file)
        print(f"Adding audio stream of {audio_clip.end} milliseconds.")
        if video_clip.end < audio_clip.end:
            audio_clip = audio_clip.subclip(0, video_clip.end)
            print(f"Clipped audio file to {video_clip.end} milliseconds.")
        elif audio_clip.end < video_clip.end:
            video_clip = video_clip.subclip(0, audio_clip.end)
            print(f"Clipped video file to {audio_clip.end} milliseconds.")

        final_video = video_clip.set_audio(audio_clip)
        print(
            f"Successfully generated video of {final_video.end} milliseconds from video and audio streams.")
        
        file_dir = os.path.dirname(video_file)
        file_name = os.path.basename(video_file).split(".")[0]
        video_out_path = f"{file_dir}/{file_name}_with_audio.mp4"
        final_video.write_videofile(video_out_path, fps=self.fps)
        print(f"Video successfully saved to {video_out_path}.")


class Visemes():
    IPA_TO_VISEME = {
        "b": "p",
        "d": "t",
        "d͡ʒ": "S",
        "ð": "T",
        "f": "f",
        "ɡ": "k",
        "h": "k",
        "j": "i",
        "k": "k",
        "l": "t",
        "m": "p",
        "n": "t",
        "ŋ": "k",
        "p": "p",
        "ɹ": "r",
        "s": "s",
        "ʃ": "S",
        "t": "t",
        "t͡ʃ": "S",
        "θ": "T",
        "v": "f",
        "w": "u",
        "z": "s",
        "ʒ": "S",
        "ə": "@",
        "ɚ": "@",
        "æ": "a",
        "aɪ": "a",
        "aʊ": "a",
        "ɑ": "a",
        "eɪ": "e",
        "ɝ": "E",
        "ɛ": "E",
        "i": "i",
        "ɪ": "i",
        "oʊ": "o",
        "ɔ": "O",
        "ɔɪ": "O",
        "u": "u",
        "ʊ": "u",
        "ʌ": "E",
    }

    def __init__(self, model_name: str = "eng2102", lang: str = "eng"):
        self.model_name = model_name
        self.lang = lang
        self.update_phone()
        self.recognizer = read_recognizer(self.model_name)

    def parse_frames(self, phones: str) -> List[Frame]:
        frames = []
        for line in phones.split("\n"):
            if line == "":
                continue
            start, duration, phone = line.split()
            viseme = self.IPA_TO_VISEME[phone]
            frames.append(Frame(float(start), float(duration), phone, viseme))
        return frames

    def fix_gaps(self, phones: List[Frame]):
        for i in range(len(phones) - 1):
            current_phone = phones[i]
            next_phone = phones[i + 1]

            max_duration = next_phone.start - current_phone.start
            current_phone.duration = min(max(current_phone.duration, 0.5), max_duration)

            if max_duration > 0.5:
                current_phone.duration = min(0.25, current_phone.duration)

    @timeit
    def recognize(self, path: str) -> List[Frame]:
        topk = 1
        emit = 1.5
        timestamp = True

        phones = self.recognizer.recognize(
            path, self.lang, topk, emit, timestamp)
        phones = self.parse_frames(phones)
        self.fix_gaps(phones)
        return phones

    def update_phone(self):
        model_path = get_model_path(self.model_name)
        inventory = Inventory(model_path)
        new_unit_file = Path(f"data/visemes/{self.lang}.txt")
        inventory.update_unit(self.lang, new_unit_file)


if __name__ == "__main__":
    wav_file = f"{CURRENT_DIR}/../../data/samples/barefoot.wav"
    out_path = f"{CURRENT_DIR}/../../data/results/visemes"
    movie_maker = VisemeMovieMaker(out_path)
    movie_maker.generate_video(wav_file)
    movie_maker.add_audio(wav_file, movie_maker.out_path)
