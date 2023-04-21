import asyncio
import json
from websockets.server import serve, WebSocketServerProtocol
from assistant_server.api_clients.gpt import gpt, SayStatement


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
    print("Client connected")

    try:
        async for message in websocket:
            message_str = message.decode("utf-8") if isinstance(message, bytes) else message
            print(f"Received message: {message_str}")
            await repeat(message_str, websocket)
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

asyncio.run(start())
