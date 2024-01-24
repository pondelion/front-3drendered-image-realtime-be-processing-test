from typing import List

from transformers import pipeline
from PIL.Image import Image as PILImage


owlvit = None


def owlvit_detect(image: PILImage, labels: List[str]):
    global owlvit
    if owlvit is None:
        owlvit = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection", device=0)
    preds = owlvit(image, candidate_labels=labels)
    # [
    #     {'score': 0.3571370542049408,
    #      'label': 'human',
    #       'box': {'xmin': 180, 'ymin': 71, 'xmax': 271, 'ymax': 178}},
    #     ...
    # ]
    return preds
