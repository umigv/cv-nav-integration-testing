import cv2
import numpy as np
from ultralytics import YOLO
import time
import paho.mqtt.client as mqtt
import json
import math
import argparse

np.bool = np.bool_
parser = argparse.ArgumentParser()
parser.add_argument("--lane_model", type=str, default="LLOnly120.pt")
parser.add_argument("--hole_model", type=str, default="potholesonly100epochs.pt")
parser.add_argument("--mqtt_ip", type=str, default="127.0.0.0")
parser.add_argument("--mqtt_port", type=int, default=1883)
parser.add_argument("--mqtt_topic", type=str, default="zed_image")
parser.add_argument("--zed", action='store_true')
parser.add_argument("--video", type=str, default="IMG_7493.mp4")
parser.add_argument("--testing", action='store_true')
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--verbose_light", action='store_true')
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()


lane_model = YOLO(args.lane_model, task='segment')
hole_model = YOLO(args.hole_model, task='detect')

def get_occupancy_grid(frame, client, MQTT_TOPIC):
    r_lane = lane_model.predict(frame, conf=0.5, device=args.device, verbose=args.verbose)[0]
    r_hole = hole_model.predict(frame, conf=0.25, device=args.device, verbose=args.verbose)[0]

    lane_data = {"masks": None}
    hole_data = {"boxes": None}

    if r_lane.masks is not None and (len(r_lane.masks.xy) != 0):
        segment = r_lane.masks.xy[0]
        lane_data["masks"] = {"xy": segment.tolist()}

    if r_hole.boxes is not None:
        hole_data["boxes"] = {"xyxyn": [segment.tolist() for segment in r_hole.boxes.xyxyn]}
        

    data = {"lane": lane_data, "hole": hole_data}
    client.publish(MQTT_TOPIC, json.dumps(data))

    if args.testing:
        time_of_frame = 0

        image_width, image_height = frame.shape[1], frame.shape[0]
        occupancy_grid = np.zeros((image_height, image_width))

        lane = data['lane']
        pothole = data['hole']

        if lane['masks'] is not None:
                if(len(lane['masks']['xy']) != 0):
                    segment = lane['masks']['xy']
                    segment_array = np.array([segment], dtype=np.int32)
                    cv2.fillPoly(occupancy_grid, [segment_array], color=(255, 0, 0))
                    time_of_frame = time.time()
        
        if pothole['boxes'] is not None:
            for segment in pothole['boxes']['xyxyn']:
                x_min, y_min, x_max, y_max = segment
                vertices = np.array([[x_min*image_width, y_min*image_height], 
                                    [x_max*image_width, y_min*image_height], 
                                    [x_max*image_width, y_max*image_height], 
                                    [x_min*image_width, y_max*image_height]], dtype=np.int32)
                cv2.fillPoly(occupancy_grid, [vertices], color=(0, 0, 0))
        
        buffer_area = np.sum(occupancy_grid)//255
        buffer_time = math.exp(-buffer_area/(image_width*image_height)-0.7)
        return occupancy_grid, buffer_time, time_of_frame
    else:
        return None, None, None


def main(testing=False):

    MQTT_BROKER = args.mqtt_ip
    MQTT_PORT = args.mqtt_port
    MQTT_TOPIC = args.mqtt_topic
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.connect(MQTT_BROKER, MQTT_PORT)

    if args.zed:
        cap = cv2.VideoCapture(0)
        if cap.isOpened() == 0:
            exit(-1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    else:
        cap = cv2.VideoCapture('IMG_7493.mp4')
    
    if testing:
        curr_time = time.time()

    memory_buffer = np.full((1280, 720), 255).astype(np.uint8)
    frame_count = 0
    total_time = 0
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not ret:
            break

        if args.zed:
            frame = np.split(frame, 2, axis=1)[0]
        else:
            frame = cv2.resize(frame, (1280, 720))

        curr_time = time.time()

        occupancy_grid_display, buffer_time, time_of_frame = get_occupancy_grid(frame, client, MQTT_TOPIC)

        if testing:
            if occupancy_grid_display is not None:
                total = np.sum(occupancy_grid_display)
                if total == 0:
                    if curr_time - time_of_frame < buffer_time:
                        occupancy_grid_display = memory_buffer
                    else:
                        occupancy_grid_display.fill(255)
                else:
                    memory_buffer = occupancy_grid_display

            occ = occupancy_grid_display.astype(np.uint8)
            overlay = cv2.addWeighted(frame, 1, cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR), 0.5, 0)
            cv2.imshow('Detection', overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if args.verbose:
            print(time.time() - curr_time)

        if args.verbose_light:
            end_time = time.time()
            frame_count += 1
            total_time += end_time - start_time
            avg_time = (total_time / frame_count) * 1000 if frame_count > 0 else 0  # Calculate average time in ms
            print(f'Processed frames: {frame_count}, Average processing time: {avg_time:.2f} ms', end='\r')

            if frame_count >= 1000000:
                frame_count = 0
                total_time = 0
                

    cap.release()
    client.disconnect()
    if testing:
        cv2.destroyAllWindows()
    exit(0)


if __name__ == '__main__':
    main(args.testing)