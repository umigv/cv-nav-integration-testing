import paho.mqtt.client as mqtt
import json

# Define Variables
MQTT_BROKER = "35.3.23.153"
MQTT_PORT = 1884
MQTT_TOPIC = "zed_image"

# Create client instance
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)

# Connect to the broker
client.connect(MQTT_BROKER, MQTT_PORT)

# Publish JSON data
data = {"key": "value"}  # replace with your JSON data
client.publish(MQTT_TOPIC, json.dumps(data))

# Disconnect from broker
client.disconnect()