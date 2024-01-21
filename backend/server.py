import asyncio
from datetime import datetime
from typing import List
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import cv2

from image_utils import decode_base64_image_url, encode_to_base64_image_url


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(len(self.active_connections))

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


class WSMessageHandler:

    def __init__(self, websocket, connection_manager):
        self._ws = websocket
        self._connection_manager = connection_manager
        self._is_connected = True
        self._image_url = None

    async def message_send_task(self) -> None:
        # prev_dt = cur_dt = datetime.now()
        while self._is_connected:
            # cur_dt = datetime.now()
            # dt = (cur_dt - prev_dt).total_seconds()
            # print('message_send_task', self._is_connected)
            if self._image_url is not None:
                # print('send')
                await self._ws.send_json({'image': self._image_url})
                # self._ws.send_json({'image': self._image_url})
            await asyncio.sleep(0.1)

    async def message_receive_task(self) -> None:
        while self._is_connected:
            msg_received = await self._ws.receive_text()
            if msg_received == 'close':
                print('received close message')
                self._connection_manager.disconnect(self._ws)
                self._is_connected = False
                break
            # print(msg_received)
            data = json.loads(msg_received)
            if 'image' in data:
                img = decode_base64_image_url(data['image'])
                if np.mean(img) != 0:
                    cv2.putText(img, "from server", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0))
                    self._image_url = encode_to_base64_image_url(img)
                # print(img.shape, np.mean(img))
            # print(f'received msg : {data}')

    @property
    def is_connected(self) -> bool:
        return self._is_connected


app = FastAPI()
manager = ConnectionManager()


@app.websocket("/image_process")
async def ws_simulate(
    websocket: WebSocket,
):
    await manager.connect(websocket)
    try:
        message_handler = WSMessageHandler(websocket, manager)

        _ = await asyncio.gather(
            message_handler.message_send_task(),
            message_handler.message_receive_task(),
        )
    except (WebSocketDisconnect, Exception) as e:
        print(e)
        manager.disconnect(websocket)
    print('end')
