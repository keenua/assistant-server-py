import asyncio
from typing import Union

import pytest
from websockets.client import connect

from assistant_server.server.echo_server import start


async def send_message(message: str) -> Union[bytes, str]:
    async with connect("ws://localhost:3000/echo") as websocket:
        await websocket.send(message)
        return await websocket.recv()


@pytest.mark.asyncio
async def test_echo_server():
    # Start the echo server
    server_task = asyncio.create_task(start())

    # Connect and interact with the server using the helper function
    message = "Hello, World!"
    response = await send_message(message)

    # Check if the response is as expected
    assert response == f"Echo: {message}"

    # Stop the server
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass
