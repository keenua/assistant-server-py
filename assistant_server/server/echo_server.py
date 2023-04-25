import asyncio

from websockets.server import serve, WebSocketServerProtocol


async def echo(websocket: WebSocketServerProtocol, path: str) -> None:
    if path != "/echo":
        await websocket.close()
        return

    async for message in websocket:
        str_message = message.decode("utf-8") if isinstance(message, bytes) else message
        print(f"Received: {str_message}")
        await websocket.send(f"Echo: {str_message}")


async def start():
    PORT = 3000
    print(f"Running server on port {PORT}")
    async with serve(echo, "", port=PORT):
        await asyncio.Future()
