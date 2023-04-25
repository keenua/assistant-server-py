from .inference import GestureInferenceModel


if __name__ == "__main__":
    model = GestureInferenceModel()

    model.load_model()
    model.infer("data/samples/barefoot.wav", "part1.bvh")
    model.infer("data/samples/barefoot.wav", "part2.bvh")