import asyncio
import logging

from websockets.server import WebSocketServerProtocol, serve

LOGGER = logging.getLogger(__name__)


async def echo(websocket: WebSocketServerProtocol, path: str) -> None:
    if path != "/echo":
        await websocket.close()
        return

    async for message in websocket:
        str_message = message.decode(
            "utf-8") if isinstance(message, bytes) else message
        LOGGER.info(f"Received: {str_message}")
        await websocket.send(f"Echo: {str_message}")


async def start():
    PORT = 3000
    LOGGER.info(f"Running server on port {PORT}")
    async with serve(echo, "", port=PORT):
        await asyncio.Future()
