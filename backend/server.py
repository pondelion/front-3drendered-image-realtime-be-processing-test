import asyncio
from datetime import datetime
from typing import List
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import cv2
from PIL import Image

from image_utils import decode_base64_image_url, encode_to_base64_image_url
from ai import owlvit_detect


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
        self._image = None
        self._seg_image = None
        self._bboxes = None

    async def message_send_task(self) -> None:
        while self._is_connected:
            # if self._image is not None:
            #     # print(datetime.now(), 'send')
            #     Image.fromarray(self._image[:, :, ::-1]).convert('RGB').save('./tmp.jpg')
            #     od_preds = owlvit_detect(Image.fromarray(self._image[:, :, ::-1]).convert('RGB'), labels=['cat'])
            #     if len(od_preds) > 0:
            #         print(od_preds)

            if self._bboxes is not None and self._image is not None:
                for bbox in self._bboxes:
                    obj_tag = bbox['obj_tag']
                    x1 = bbox['bbox']['top_left_x']
                    y1 = bbox['bbox']['top_left_y']
                    x2 = bbox['bbox']['bottom_right_x']
                    y2 = bbox['bbox']['bottom_right_y']
                    if all([0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)]):
                        y1 = -y1 + 1.0
                        y2 = -y2 + 1.0
                        x1 = int(self._image.shape[1] * x1)
                        y1 = int(self._image.shape[0] * y1)
                        x2 = int(self._image.shape[1] * x2)
                        y2 = int(self._image.shape[0] * y2)
                        cv2.rectangle(
                            self._image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0),
                            thickness=3, lineType=cv2.LINE_4,shift=0
                        )
            if self._image is not None:
                # print(datetime.now(), 'send')
                cv2.putText(self._image, f"from server {datetime.now()}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0))
                self._image_url = encode_to_base64_image_url(self._image)
                params = {'image': self._image_url}
                if self._seg_image is not None:
                    cv2.putText(self._seg_image, f"from server {datetime.now()}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0))
                    seg_image_url = encode_to_base64_image_url(self._seg_image)
                    params['seg_image'] = seg_image_url
                await self._ws.send_json(params)
                # self._ws.send_json({'image': self._image_url})
                await asyncio.sleep(0.15)
            else:
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
                # print(datetime.now(), 'recv')
                org_img = decode_base64_image_url(data['image'])
                if np.mean(org_img) == 0:
                    continue
                if 'seg_image' in data:
                    seg_img = decode_base64_image_url(data['seg_image'])
                    green_mask = (
                        (seg_img[:, :, 0:1] == 0)
                        &
                        (seg_img[:, :, 1:2] == 255)
                        &
                        (seg_img[:, :, 2:3] == 0)
                    )
                    self._seg_image = org_img * green_mask

                self._image = org_img
                if 'bboxes' in data:
                    self._bboxes = data['bboxes']
                # cv2.putText(self._image, f"from server {datetime.now()}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0))
                self._image_url = encode_to_base64_image_url(self._image)
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
