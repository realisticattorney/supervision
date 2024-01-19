import numpy as np  # this is numpy, a library for working with arrays
import supervision as sv  # this is the supervision library. Supervision in the context of ML & computer vision refers to the use of labeled data to train models
from ultralytics import (
    YOLO,
)  # this is the ultralytics library, which contains the YOLO model,a neural network that can detect objects in images

model = YOLO("yolov8n.pt")
box_annotator = sv.BoundingBoxAnnotator()


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    return box_annotator.annotate(frame.copy(), detections=detections)


sv.process_video(
    source_path="inputvids/resumen_fulbo.mp4",
    target_path="outputvids/resultado.mp4",
    callback=callback,
)



#GOALS:
# 1) Detect players and ball
# 2) Detect the field lines and measure distances so we can calculate the size of the field + the size of the players + the speed of the players and the ball

    