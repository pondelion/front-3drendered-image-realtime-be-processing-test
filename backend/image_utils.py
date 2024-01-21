import base64

import cv2
import numpy as np


def decode_base64_image_url(img_url: str) -> np.ndarray:
    header, encoded = img_url.split(',', 1)
    img = base64.b64decode(encoded)
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img


def encode_to_base64_image_url(image: np.ndarray) -> str:
    header = 'data:image/png;base64,'
    png_img = cv2.imencode('.png', image)
    b64_img_str = base64.b64encode(png_img[1]).decode('utf-8')
    image_url = f'{header}{b64_img_str}'
    return image_url
