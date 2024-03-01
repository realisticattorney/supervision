import argparse
import cv2
import supervision as sv
import numpy as np
from ultralytics import (
    YOLO,
)
import torch

from collections import defaultdict, deque

# SOURCE_2k_first_15mins = np.array([[774, 555], [2713, 689], [2357, 922], [-464, 641]])
SOURCE = np.array(
    [[2649, 1203], [5141, 1350], [2955, 2581], [236, 1382]]
)  # _4k_last_20mins

TARGET_WIDTH = 1790  # cm
TARGET_HEIGHT = 2270

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a video")
    parser.add_argument(
        "--source_video_path",
        type=str,
        required=True,
        help="Path to the video to process",
    )
    parser.add_argument(
        "--output_video_path",
        type=str,
        default=None,
        help="Path to save the output video with annotations, disables display if set.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    # model = YOLO("yolov5xu.pt")
    model = torch.hub.load("ultralytics/yolov5", "custom", path="./best.pt")

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    scaled_resolution_wh = (
        video_info.resolution_wh[0] // 2,
        video_info.resolution_wh[1] // 2,
    )

    thickness = sv.calculate_dynamic_line_thickness(resolution_wh=scaled_resolution_wh)
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=scaled_resolution_wh)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)

    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    frame_generator = sv.get_video_frames_generator(
        args.source_video_path
    )  # creates an instance of FrameGenerator to look over frames of our input video

    polygon_zone = sv.PolygonZone(SOURCE, frame_resolution_wh=video_info.resolution_wh)
    view_transformer = ViewTransformer(SOURCE, TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    display_enabled = args.output_video_path is None

    if args.output_video_path:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
        out = cv2.VideoWriter(
            args.output_video_path,
            fourcc,
            video_info.fps,
            (video_info.resolution_wh[0], video_info.resolution_wh[1]),
        )

    for frame in frame_generator:
        # result = model(frame)[0]
        # Convert the frame from BGR (OpenCV default) to RGB (expected by the model)
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # # Perform inference, specifying the size for resizing the input images
        # results = model([frame_rgb], size=640)
        # # Extract the detection results for the first image in the batch
        # detections = results.xyxy[0]

        result = model(frame)
        detections = sv.Detections.from_yolov5(result)

        # formatted_detections = []
        # for det in detections:
        #     x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        #     formatted_detection = {"bbox": [x1, y1, x2, y2], "conf": conf, "cls": cls}
        #     formatted_detections.append(formatted_detection)

        # detections = sv.Detections.from_ultralytics(result)
        detections = sv.Detections.from_yolov5(detections)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        scaling_factor = TARGET_HEIGHT / np.linalg.norm(TARGET[2] - TARGET[0])
        labels = []
        for tracker_id, [x, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append((x, y))
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                x1, y1 = coordinates[tracker_id][0]
                x2, y2 = coordinates[tracker_id][-1]

                # Calculate pixel distance using Euclidean formula
                distance_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                # Convert distance to cm using the scaling factor
                distance_cm = distance_px * scaling_factor
                # Calculate speed; time elapsed is 1/30 of a second for each frame, but using the whole deque (0.5 seconds here)
                speed = (
                    distance_cm * 0.036 / (len(coordinates[tracker_id]) / 30)
                )  # km/h
                labels.append(f"#{tracker_id} {speed:.2f} km/h")

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(
            annotated_frame,
            polygon=SOURCE,
            color=sv.Color.red(),
        )
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        if args.output_video_path:
            out.write(annotated_frame)  # Write the frame to the output video

        if display_enabled:
            cv2.imshow("annotated_frame", annotated_frame)
            if cv2.waitKey(1) == ord("q"):
                break
    if args.output_video_path:
        out.release()  # Release the VideoWriter object

    if display_enabled:
        cv2.destroyAllWindows()  # Close all OpenCV windows
