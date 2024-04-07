import gi
import paho.mqtt.client as mqtt
import sys
from gi.repository import Gst, GObject, GLib
import pyds
import json

gi.require_version('Gst', '1.0')

Gst.init(sys.argv)

loop = GLib.MainLoop()

# MQTT setup
broker = "localhost"
port = 1883
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) 
client.connect(broker, port)

print("Connected to MQTT broker")

def bus_call(bus, message, data):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End of stream\n")
        client.disconnect()
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write(f"Error: {err}, {debug}\n")
        client.disconnect()
        loop.quit()
    return True

pipeline = Gst.Pipeline()

source = Gst.ElementFactory.make('zedsrc', 'zed-source')
source.set_property('stream-type', 0)  # 2D left image

capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
caps = Gst.Caps.from_string("video/x-raw, format=(string)RGBA")
capsfilter.set_property("caps", caps)


streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
streammux.set_property('width', 1280)
streammux.set_property('height', 720)
streammux.set_property('batch-size', 1)
streammux.set_property('batched-push-timeout', 4000000)

nvinfer = Gst.ElementFactory.make("nvinfer", "primary-inference")
nvinfer.set_property('config-file-path', 'config_infer.txt')

pipeline.add(source)
pipeline.add(capsfilter)
pipeline.add(streammux)
pipeline.add(nvinfer)

source.link(capsfilter)
capsfilter.link(streammux)
streammux.link(nvinfer)

def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK
    
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            continue

        l_obj = frame_meta.obj_meta_list

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                continue

            print(f"Object {obj_meta.class_id} detected at ({obj_meta.rect_params.left}, {obj_meta.rect_params.top})")

            oj_info = {}

            try:
                client.publish("topic", json.dumps(oj_info))
            except BrokenPipeError:
                print("Connection closed")
                loop.quit()
                break
            
            l_obj = l_obj.next

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK

bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", bus_call, client)

pipeline.set_state(Gst.State.PLAYING)

try:
    loop.run()
except Exception as e:
    print(e)
    pass
finally:
    pipeline.set_state(Gst.State.NULL)
    client.disconnect()

