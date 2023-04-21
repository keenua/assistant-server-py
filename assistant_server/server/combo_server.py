import asyncio
import json
import os

from websockets.server import WebSocketServerProtocol, serve

from assistant_server.api_clients.gpt import SayStatement, gpt
from assistant_server.api_clients.speech import generate_speech
from assistant_server.gesture_generation.inference import GestureInferenceModel
from assistant_server.server.utils import bytes_to_base64, mp3_to_wav

model = GestureInferenceModel()
model.load_model()


async def process(prompt: str, websocket: WebSocketServerProtocol) -> None:
    async for statement in gpt(prompt):
        if isinstance(statement, SayStatement):
            data = statement.__dict__.copy()

            audio = generate_speech(statement.text)

            wav_file = "temp.wav"
            mp3_to_wav(audio, wav_file)

            data["audio"] = bytes_to_base64(audio)
            data["bvh"] = model.infer(wav_file)

            os.remove(wav_file)

            await websocket.send(json.dumps(data))


async def handle_connection(websocket: WebSocketServerProtocol, path: str) -> None:
    print("Client connected")

    try:
        async for message in websocket:
            message_str = message.decode(
                "utf-8") if isinstance(message, bytes) else message
            print(f"Received message: {message_str}")
            await process(message_str, websocket)
            print("GPT done")

    except Exception as error:
        print(f"GPT error: {error}")

    finally:
        print("Client disconnected")


async def start():
    port = 3000
    print(f"WebSocket server is running on port {port}")
    async with serve(handle_connection, "localhost", port):
        await asyncio.Future()

if __name__ == "__main__":
    text = "Hello, my name is Rachel."
    audio = generate_speech(text)

    wav_file = "temp.wav"
    mp3_to_wav(audio, wav_file)

    data = {}
    data["text"] = text
    data["audio"] = bytes_to_base64(audio)
    data["bvh"] = model.infer(wav_file)

    os.remove(wav_file)

    with open("temp.json", "w") as file:
        json.dump(data, file)
