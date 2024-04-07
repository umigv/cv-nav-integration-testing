import paho.mqtt.client as mqtt

# Define Variables
MQTT_BROKER = "35.3.23.153"
MQTT_PORT = 1884
MQTT_TOPIC = "zed_image"

# Create client instance
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    print(f"Topic: {msg.topic}\nMessage: {msg.payload.decode()}")

client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT)

client.loop_forever()