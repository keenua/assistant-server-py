import asyncio
import json
import logging

from websockets.server import WebSocketServerProtocol, serve

from assistant_server.api_clients.gpt import SayStatement, gpt

LOGGER = logging.getLogger(__name__)

async def read_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


async def test(websocket: WebSocketServerProtocol) -> None:
    user_prompt = await read_file("./sample_prompt.txt")
    async for statement in gpt(user_prompt):
        if isinstance(statement, SayStatement):
            await websocket.send(json.dumps(statement.__dict__))


async def repeat(prompt: str, websocket: WebSocketServerProtocol) -> None:
    async for statement in gpt(prompt):
        if isinstance(statement, SayStatement):
            await websocket.send(json.dumps(statement.__dict__))


async def handle_connection(websocket: WebSocketServerProtocol, path: str) -> None:
    LOGGER.info("Client connected")

    try:
        async for message in websocket:
            message_str = message.decode("utf-8") if isinstance(message, bytes) else message
            LOGGER.info(f"Received message: {message_str}")
            await repeat(message_str, websocket)
            LOGGER.info("GPT done")

    except Exception as error:
        LOGGER.error(f"GPT error: {error}")

    finally:
        LOGGER.info("Client disconnected")


async def start():
    port = 3000
    LOGGER.info(f"WebSocket server is running on port {port}")
    async with serve(handle_connection, "localhost", port):
        await asyncio.Future()

asyncio.run(start())
