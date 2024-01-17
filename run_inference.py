import numpy as np
import supervision as sv
from ultralytics import YOLO

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
