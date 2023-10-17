from ultralytics import YOLO
import cv2
import argparse
import supervision as sv

def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P_E_D_R_O arguments")
    parser.add_argument(
        "--camera-size",
        default=[1280, 720],
        nargs=2,
        type=int
    ) 
    args = parser.parse_args()
    return args


def main():
    args = parse_argument()
    width, height = args.camera_size

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    model = YOLO("/Users/pedrohenriquediashemmeldeoliveirasouza/Documents/Github/P_E_D_R_O-classifier/trained-model/best.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()

        results = model(frame)[0]

        detections = sv.Detections.from_yolov8(results)
        labels = [
            f"{model.model.names[int(class_id)]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        cv2.imshow("P_E_D_R_O pounds classifier", frame)

        if (cv2.waitKey(30) == 27):
            break;


if __name__ == "__main__":
    main()