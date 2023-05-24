from dataclasses import dataclass
from pathlib import Path
from typing import List

from allosaurus.app import read_recognizer
from allosaurus.lm.inventory import Inventory
from allosaurus.model import get_model_path

from assistant_server.gesture_generation.utils import timeit


@dataclass
class Frame:
    start: float
    duration: float
    phone: str
    viseme: str


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

    @timeit
    def recognize(self, path: str) -> List[Frame]:
        topk = 1
        emit = 0.7
        timestamp = True

        phones = self.recognizer.recognize(
            path, self.lang, topk, emit, timestamp)
        phones = self.parse_frames(phones)
        return phones

    def update_phone(self):
        model_path = get_model_path(self.model_name)
        inventory = Inventory(model_path)
        new_unit_file = Path(f"data/visemes/{self.lang}.txt")
        inventory.update_unit(self.lang, new_unit_file)


if __name__ == "__main__":
    generator = Visemes()
    phones = generator.recognize("./data/samples/barefoot.wav")
    print(phones)
