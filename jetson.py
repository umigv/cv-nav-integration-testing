import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
import paho.mqtt.client as mqtt
import json
import threading
from multiprocessing import Process, Queue

lane_model = YOLO("LLOnly180ep.pt")
hole_model = YOLO('potholesonly100epochs.pt')

def publish(client, topic, message):
    client.publish(topic, json.dumps(message.tolist()))

def predict_lane_model(frame, queue):
    r_lane = lane_model.predict(frame, conf=0.5, device='mps')[0].cpu()
    queue.put(r_lane)

def predict_hole_model(frame, queue):
    r_hole = hole_model.predict(frame, conf=0.25, device='mps')[0].cpu()
    queue.put(r_hole)

def get_occupancy_grid(frame):
        
        r_lane = lane_model.predict(frame, conf=0.5, device='mps')[0].cpu()

        # lane_annotated_frame = r_lane.plot()
        image_width, image_height = frame.shape[1], frame.shape[0]
        
        occupancy_grid = np.zeros((image_height, image_width))
        r_hole = hole_model.predict(frame, conf=0.25, device='mps')[0].cpu()
        time_of_frame = 0
        if r_lane.masks is not None:
            if(len(r_lane.masks.xy) != 0):
                with open("lane_detection_xy.txt", "w") as f:
                    f.write(str(r_lane.masks.xy))
                segment = r_lane.masks.xy[0]
                segment_array = np.array([segment], dtype=np.int32)
                cv2.fillPoly(occupancy_grid, [segment_array], color=(255, 0, 0))
                time_of_frame = time.time()

        if r_hole.boxes is not None:
            for segment in r_hole.boxes.xyxyn:
                x_min, y_min, x_max, y_max = segment
                vertices = np.array([[x_min*image_width, y_min*image_height], 
                                    [x_max*image_width, y_min*image_height], 
                                    [x_max*image_width, y_max*image_height], 
                                    [x_min*image_width, y_max*image_height]], dtype=np.int32)
                cv2.fillPoly(occupancy_grid, [vertices], color=(0, 0, 0))

        buffer_area = np.sum(occupancy_grid)//255
        buffer_time = math.exp(-buffer_area/(image_width*image_height)-0.7)
        return occupancy_grid, buffer_time, time_of_frame


def main():

    MQTT_BROKER = "35.3.157.95"
    MQTT_PORT = 1883
    MQTT_TOPIC = "zed_image"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(MQTT_BROKER, MQTT_PORT)


    cap = cv2.VideoCapture('IMG_7493.mp4')

    out = None
    
    curr_time = time.time()
    memory_buffer = np.full((1280, 720), 255).astype(np.uint8)
    while True:
        # Read in an image:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        occupancy_grid_display, buffer_time, time_of_frame = get_occupancy_grid(frame)
        occ = occupancy_grid_display.astype(np.uint8)
        overlay = cv2.addWeighted(frame, 1, cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR), 0.5, 0)
        total = np.sum(occupancy_grid_display)
        curr_time = time.time()
        if total == 0:
            if curr_time - time_of_frame < buffer_time:
                occupancy_grid_display = memory_buffer
            else:
                occupancy_grid_display.fill(255)
        else:
            memory_buffer = occupancy_grid_display
        
        publish_thread = threading.Thread(target=publish, args=(client, MQTT_TOPIC, occupancy_grid_display))
        publish_thread.start()
        # client.publish(MQTT_BROKER, json.dumps(occupancy_grid_display.tolist()))
        print(time.time() - curr_time)


        cv2.imshow('Original Video', overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    client.disconnect()


if __name__ == '__main__':
    main()