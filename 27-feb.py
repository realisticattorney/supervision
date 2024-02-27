import argparse

import supervision as sv

# from inference.models.utils import get_roboflow_model
from ultralytics import (
    YOLO,
)  # this is the ultralytics library, which contains the YOLO model,a neural network that can detect objects in images

model = YOLO("yolov5xu.pt")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a video")
    parser.add_argument(
        "--source_video_path",
        type=str,
        required=True,
        help="Path to the video to process",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # creates an instance of FrameGenerator to look over frames of our input video
    frame_generator = sv.get_video_frames_generator(args.source_video_path)
    for frame in frame_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        