# CV Jetson Scripts

This script is used for lane segmentation and pothole detection in video streams. It uses the YOLO model for object detection and segmentation.

## Requirements

- Python 3.6 or above
- OpenCV
- Ultralytics
- Paho MQTT
- Numpy

## Getting the Models to TRT

To convert the models to TensorRT, use the following command:

```bash
python convert-trt.py --lane_model my_lane_model.pt --hole_model my_hole_model.pt
```

This will generate the TensorRT models in the working directory.

> [!NOTE]  
> This can take a while to run, so be patient. On the Orin each model can take more than 10 minutes to convert.



## Command Line Arguments

The script accepts the following command line arguments:

- `--lane_model`: The name of the lane segmentation model (default: "LLOnly120.pt") it is recommended to use the TRT model
- `--hole_model`: The name of the pothole detection model (default: "potholesonly100epochs.pt") it is recommended to use the TRT model
- `--mqtt_ip`: The IP address of the MQTT broker (default: "127.0.0.0")
- `--mqtt_port`: The port number of the MQTT broker (default: 1883)
- `--mqtt_topic`: The MQTT topic to publish the data to (default: "zed_image")
- `--zed`: Use this flag to use the ZED camera for video input
- `--video`: The path to the video file to use for input (default: "IMG_7493.mp4")
- `--testing`: Use this flag to enable testing mode
- `--verbose`: Use this flag to enable verbose output
- `--verbose_light`: Use this flag to enable light verbose output which only prints average processing time and # of frames processed
- `--device`: The device to use for model inference (default: "cuda"), can be "cuda" or "cpu" or in some cases "mps"

## Usage

To run the script with default arguments, use the following command:

```bash
python jetson.py
```

To specify arguments, use the following command:

```bash
python jetson.py --lane_model my_lane_model.pt --hole_model my_hole_model.pt --mqtt_ip 127.0.0.0 --mqtt_port 1884 --mqtt_topic my_topic --video my_video.mp4
```

When running the script for testing ie. the startup script should not be run, use the following command:

```bash
python jetson.py --lane_model /full/path/to/my_lane_model.engine --hole_model /full/path/to/my_hole_model.engine --zed
```

## Output

The script processes each frame of the video input, performs lane segmentation and pothole detection, and publishes the results to the specified MQTT topic which is then subscribed by the ROS node for occupancy grid mapping.