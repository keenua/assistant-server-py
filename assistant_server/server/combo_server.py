import asyncio
import json
import os
from typing import Awaitable, Callable, List

from websockets.server import WebSocketServerProtocol, serve

from assistant_server.api_clients.gpt import SayStatement, gpt
from assistant_server.server.director import Director, Frame, StatementData


async def process(prompt: str, director: Director) -> None:
    async for statement in gpt(prompt):
        if isinstance(statement, SayStatement):
            statement_data = StatementData(statement.text, statement.emotion)
            director.add_statement(statement_data)


async def send_frames(director: Director, on_frames: Callable[[List[Frame]], Awaitable[None]]) -> None:
    while True:
        frames = director.get_frames()

        if frames:
            await on_frames(frames)
        else:
            await asyncio.sleep(0.1)


async def handle_connection(websocket: WebSocketServerProtocol, path: str) -> None:
    print("Client connected")
    director = Director()
    director.start()

    async def process_frames(frames: List[Frame]):
        print(f"Sending {len(frames)} frames")
        result = {
            "frames": {frame.index: frame.__dict__ for frame in frames}
        }
        await websocket.send(json.dumps(result))

    try:
        asyncio.create_task(send_frames(director, process_frames))

        while True:
            async for message in websocket:
                message_str = message.decode(
                    "utf-8") if isinstance(message, bytes) else message
                print(f"Received message: {message_str}")
                await process(message_str, director)
                print("GPT done")

    except Exception as error:
        print(f"GPT error: {error}")

    finally:
        print("Client disconnected")


async def start():
    port = 3000
    print(f"WebSocket server is running on port {port}")
    async with serve(handle_connection, port=port):
        await asyncio.Future()

if __name__ == "__main__":
    index: int = 0
    current_dir = os.path.dirname(os.path.realpath(__file__))

    async def log_frames(frames: List[Frame]):
        global index
        with open(f"{current_dir}/../../data/results/temp{index}.json", "w") as file:
            data = [frame.__dict__ for frame in frames]
            json.dump(data, file)
        index += 1

    async def main():
        text = "Recite a famous Al Pacino's monologue"
        director = Director()
        director.start()

        asyncio.create_task(send_frames(director, on_frames=log_frames))

        await process(text, director)

        while not director.is_idle():
            await asyncio.sleep(0.1)

        await asyncio.sleep(4)

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
