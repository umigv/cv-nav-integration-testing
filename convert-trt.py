from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lane_model", type=str, default="LLOnly120.pt")
parser.add_argument("--hole_model", type=str, default="potholesonly100epochs.pt")
args = parser.parse_args()

# Latest version of Lane Segmentation model
lane = YOLO(args.lane_model)

# Latest version of Pothole Detection model
hole = YOLO(args.hole_model)

# Convert the above models to TensorRT
lane.export(format='engine', half=True)

print("Lane model converted to TensorRT")

hole.export(format='engine', half=True)

print("Pothole model converted to TensorRT")