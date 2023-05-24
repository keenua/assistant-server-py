import asyncio
import json
import os
import time
from typing import Awaitable, Callable, List, Optional

from websockets.server import WebSocketServerProtocol, serve
from websockets.exceptions import ConnectionClosed

from assistant_server.api_clients.gpt import SayStatement, gpt
from assistant_server.server.director import Director, Frame, StatementData

last_ping_time = time.time()
frame_package_index: int = 0
current_dir = os.path.dirname(os.path.realpath(__file__))
frames_dir = f"{current_dir}/../../data/results/frames"

FRAMES_FROM_FILE = False

async def process_prompt(prompt: str, director: Director) -> None:
    async for statement in gpt(prompt):
        if isinstance(statement, SayStatement):
            statement_data = StatementData(statement.text, statement.emotion)
            director.add_statement(statement_data)


async def process_queue(queue: asyncio.Queue[str], director: Director) -> None:
    while True:
        prompt = await queue.get()
        if not FRAMES_FROM_FILE:
            await process_prompt(prompt, director)


async def send_frames(director: Director, on_frames: Callable[[List[Frame]], Awaitable[None]]) -> None:
    while True:
        frames = director.get_frames()

        if frames:
            await on_frames(frames)
        else:
            await asyncio.sleep(0.1)


async def check_heartbeat(disconnect_timeout=6):
    global last_ping_time
    last_ping_time = time.time()

    while True:
        await asyncio.sleep(1)
        if time.time() - last_ping_time > disconnect_timeout:
            print("Ping timeout, raising close exception")
            raise ConnectionClosed(None, None)


async def handle_messages(websocket: WebSocketServerProtocol, processing_queue: asyncio.Queue[str]):
    while True:
        try:
            message = await websocket.recv()

            message_str = message.decode(
                "utf-8") if isinstance(message, bytes) else message
            print(f"Received message: {message_str}")

            if message_str == "ping":
                global last_ping_time
                last_ping_time = time.time()
            else:
                processing_queue.put_nowait(message_str)
        except ConnectionClosed:
            print("Connection closed, stopping message handler")
            break


async def handle_connection(websocket: WebSocketServerProtocol, path: str) -> None:
    print("Client connected")

    director = Director()
    director.start(frames_dir if FRAMES_FROM_FILE else None)
    
    async def process_frames(frames: List[Frame]):
        global frame_package_index

        print(f"Sending {len(frames)} frames")
        result = {
            "frames": [frame.__dict__ for frame in frames]
        }

        with open(f"{current_dir}/../../data/results/frames/temp{frame_package_index}.json", "w") as file:
            data = [frame.__dict__ for frame in frames]
            json.dump(data, file)
        frame_package_index += 1
        try:
            await websocket.send(json.dumps(result))
        except ConnectionClosed as e:
            print("Connection closed")
            raise e

    processing_queue: asyncio.Queue[str] = asyncio.Queue()
    process_queue_task: Optional[asyncio.Task] = None
    handle_messages_task: Optional[asyncio.Task] = None
    send_frames_task: Optional[asyncio.Task] = None
    heartbeat_task: Optional[asyncio.Task] = None

    try:
        process_queue_task = asyncio.ensure_future(
            process_queue(processing_queue, director))

        send_frames_task = asyncio.ensure_future(
            send_frames(director, process_frames))

        handle_messages_task = asyncio.ensure_future(
            handle_messages(websocket, processing_queue))

        heartbeat_task = asyncio.ensure_future(
            check_heartbeat(10))

        if heartbeat_task:
            heartbeat_task.add_done_callback(
                lambda _: handle_messages_task and handle_messages_task.cancel())

        print("Waiting for messages")

        await handle_messages_task

    finally:
        print("Client disconnected")

        if process_queue_task:
            process_queue_task.cancel()
        if send_frames_task:
            send_frames_task.cancel()
        if heartbeat_task:
            heartbeat_task.cancel()

        director.stop()


async def start():
    port = 3000
    print(f"WebSocket server is running on port {port}")
    async with serve(handle_connection, port=port):
        await asyncio.Future()

if __name__ == "__main__":
    async def log_frames(frames: List[Frame]):
        global frame_package_index
        with open(f"{current_dir}/../../data/results/local/temp{frame_package_index}.json", "w") as file:
            data = [frame.__dict__ for frame in frames]
            json.dump(data, file)
        frame_package_index += 1

    async def main():
        text = "Recite a famous Al Pacino's monologue"
        director = Director()
        director.start()

        send_frames_task = asyncio.ensure_future(send_frames(director, log_frames))

        await process_prompt(text, director)

        while not director.is_idle():
            await asyncio.sleep(0.1)

        await asyncio.sleep(4)
        send_frames_task.cancel()

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
