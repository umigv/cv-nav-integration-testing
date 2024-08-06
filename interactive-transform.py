import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from math import radians, cos, degrees

class CameraProperties(object):
    functional_limit = radians(64.0)
    def __init__(self, height, fov_vert, fov_horz, cameraTilt):
        self.height = float(height)
        self.fov_vert = radians(float(fov_vert))
        self.fov_horz = radians(float(fov_horz))
        self.cameraTilt = radians(float(cameraTilt))
        self.bird_src_quad = None
        self.bird_dst_quad = None
        self.matrix = None
        self.maxHeight = None
        self.maxWidth = None
        self.minIndex = None

    def src_quad(self, rows, columns):
        if self.bird_src_quad is None:
            self.bird_src_quad = np.array([[0, rows - 1], [columns - 1, rows - 1], [0, 0], [columns - 1, 0]], dtype = 'float32')
        return self.bird_src_quad

    def dst_quad(self, rows, columns, min_angle, max_angle):
        if self.bird_dst_quad is None:
            fov_offset = self.cameraTilt - self.fov_vert/2.0
            bottom_over_top = cos(max_angle + fov_offset)/cos(min_angle + fov_offset)
            bottom_width = columns*bottom_over_top
            blackEdge_width = (columns - bottom_width)/2
            leftX = blackEdge_width
            rightX = leftX + bottom_width
            self.bird_dst_quad = np.array([[leftX, rows], [rightX, rows], [0, 0], [columns, 0]], dtype = 'float32')
        return self.bird_dst_quad

    def reset(self):
        self.bird_src_quad = None
        self.bird_dst_quad = None
        self.matrix = None
        self.maxHeight = None
        self.maxWidth = None
        self.minIndex = None

    def compute_min_index(self, rows, max_angle):
        self.minIndex = int(rows*(1.0 - max_angle/self.fov_vert))
        return self.minIndex

    def compute_max_angle(self):
        return min(CameraProperties.functional_limit - self.cameraTilt + self.fov_vert/2.0, self.fov_vert)

def getBirdView(image, cp):
    rows, columns = image.shape[:2]
    if columns == 1280:
        columns = 1344
    if rows == 720:
        rows = 752
    min_angle = 0.0
    max_angle = cp.compute_max_angle()
    min_index = cp.compute_min_index(rows, max_angle)
    image = image[min_index:, :]
    rows = image.shape[0]

    src_quad = cp.src_quad(rows, columns)
    dst_quad = cp.dst_quad(rows, columns, min_angle, max_angle)
    warped, bottomLeft, bottomRight, topRight, topLeft = perspective(image, src_quad, dst_quad, cp)
    return warped, bottomLeft, bottomRight, topRight, topLeft, cp.maxWidth, cp.maxHeight

def perspective(image, src_quad, dst_quad, cp):
    bottomLeft, bottomRight, topLeft, topRight = dst_quad
    widthA = topRight[0] - topLeft[0]
    widthB = bottomRight[0] - bottomLeft[0]
    maxWidth1 = max(widthA, widthB)
    heightA = bottomLeft[1] - topLeft[1]
    heightB = bottomRight[1] - topRight[1]
    maxHeight1 = max(heightA, heightB)

    matrix1 = cv2.getPerspectiveTransform(src_quad, dst_quad)
    cp.matrix = matrix1
    cp.maxWidth = int(maxWidth1)
    cp.maxHeight = int(maxHeight1)
    warped = cv2.warpPerspective(image, matrix1, (cp.maxWidth, cp.maxHeight))


    return warped, bottomLeft, bottomRight, topRight, topLeft


def calc_distance(p1, p2):
    return int(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

class Application(tk.Frame):
    def __init__(self, master=None, cp=None):
        super().__init__(master)
        self.master = master
        self.cp = cp
        self.pack()
        self.create_widgets()
        self.points = []
        self.oval_id = []
        self.update_image()

    def create_widgets(self):
        # Create controls for camera properties
        self.controls_frame = tk.Frame(self)
        self.controls_frame.pack()

        # Add the labels and entries to the frame using grid
        self.height_label = tk.Label(self.controls_frame, text="Height")
        self.height_label.grid(row=0, column=0)
        self.height_var = tk.StringVar(value=str(self.cp.height))
        self.scale_height = tk.Entry(self.controls_frame, textvariable=self.height_var)
        self.scale_height.grid(row=1, column=0)

        self.fov_vert_label = tk.Label(self.controls_frame, text="Vertical Field of View")
        self.fov_vert_label.grid(row=0, column=1)
        self.fov_vert_var = tk.StringVar(value=str(degrees(self.cp.fov_vert)))
        self.scale_fov_vert = tk.Entry(self.controls_frame, textvariable=self.fov_vert_var)
        self.scale_fov_vert.grid(row=1, column=1)

        self.fov_horz_label = tk.Label(self.controls_frame, text="Horizontal Field of View")
        self.fov_horz_label.grid(row=0, column=2)
        self.fov_horz_var = tk.StringVar(value=str(degrees(self.cp.fov_horz)))
        self.scale_fov_horz = tk.Entry(self.controls_frame, textvariable=self.fov_horz_var)
        self.scale_fov_horz.grid(row=1, column=2)

        self.camera_tilt_label = tk.Label(self.controls_frame, text="Camera Tilt")
        self.camera_tilt_label.grid(row=0, column=3)
        self.camera_tilt_var = tk.StringVar(value=str(round(degrees(self.cp.cameraTilt), 5)))
        self.scale_camera_tilt = tk.Entry(self.controls_frame, textvariable=self.camera_tilt_var)
        self.scale_camera_tilt.grid(row=1, column=3)

        self.set_button = tk.Button(self)
        self.set_button["text"] = "Set Camera Properties"
        self.set_button["command"] = self.update_image
        self.set_button.pack(side="top")


        self.canvas = tk.Canvas(self, width=1280, height=400)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.record_point)

        self.distance_label = tk.Label(self)
        self.distance_label.pack()

        self.distance_curr = tk.Label(self)
        self.distance_curr.pack()

        self.save_button = tk.Button(self)
        self.save_button["text"] = "Save Bird's Eye View"
        self.save_button.pack(side="top")

    def record_point(self, event):
        self.points.append((event.x, event.y))
        oval_id =  self.canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill="blue")
        self.oval_id.append(oval_id)

        if len(self.points) == 2:
            distance = calc_distance(*self.points)
            self.distance_label["text"] = f"Distance: {distance} pixels"
            self.distance_curr["text"] = f"Current Pixel Size: {round(0.3048 / distance, 6)} meters"
            self.points.clear()

        if len(self.oval_id) > 2:
            self.canvas.delete(self.oval_id.pop(0))
            self.canvas.delete(self.oval_id.pop(0))

    def update_image(self):
        height = self.scale_height.get()
        fov_vert = self.scale_fov_vert.get()
        fov_horz = self.scale_fov_horz.get()
        camera_tilt = self.scale_camera_tilt.get()

        ZED = CameraProperties(height, fov_vert, fov_horz, camera_tilt)

        img_path = "test2.png"
        img = cv2.imread(img_path)

        split = img.shape[1] // 2
        left_img = img[:, :split]

        bird_view, bottomLeft, bottomRight, topRight, topLeft, maxWidth, maxHeight = getBirdView(left_img, ZED)

        self.master.geometry(f"{bird_view.shape[1]}x{bird_view.shape[0] + 300}")
        self.canvas.config(width=(bird_view.shape[1]) , height=bird_view.shape[0])

        cv2.putText(bird_view, f"Height: {height}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(bird_view, f"Vertical FOV: {fov_vert}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(bird_view, f"Horizontal FOV: {fov_horz}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(bird_view, f"Camera Tilt: {camera_tilt}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        self.image = Image.fromarray(cv2.cvtColor(bird_view, cv2.COLOR_BGR2RGB))
        self.image_tk = ImageTk.PhotoImage(self.image)

        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

        self.save_button["command"] = self.save_image

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            self.image.save(file_path)

root = tk.Tk()
root.title("Bird's Eye View")
app = Application(master=root, cp=CameraProperties(64, 68.0, 101.0, 60.0))
app.mainloop()