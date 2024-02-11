import cv2
import numpy as np
from math import radians, cos
from ultralytics import YOLO

model = YOLO("good.pt")

class CameraProperties(object):
    functional_limit = radians(70.0)
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
    if (cp.matrix is None):
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
        return perspective(image, src_quad, dst_quad, cp)
    else:
        image = image[cp.minIndex:, :]
        return cv2.warpPerspective(image, cp.matrix, (cp.maxWidth, cp.maxHeight))

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
    return cv2.warpPerspective(image, matrix1, (cp.maxWidth, cp.maxHeight))

point1 = None
point2 = None

def click_event(event, x, y, flags, params):
    # Referencing global variables 
    global point1, point2

    # Checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        if point1 is None:  # if the first point is not selected, select it
            point1 = (x, y)
            cv2.circle(params, point1, 5, (0, 255, 0), -1)  # draw a green circle at point1
        elif point2 is None:  # if the first point is selected and the second is not, select the second
            point2 = (x, y)
            cv2.circle(params, point2, 5, (0, 0, 255), -1)  # draw a red circle at point2
def main():
    global point1, point2

    ZED = CameraProperties(54, 68.0, 101.0, 50.0)

    #ZED = CameraProperties(54, 68.0, 101.0, 68.0)
    # image = cv2.imread('test-new.png')
    # transformed_image = getBirdView(image, ZED)

    cap = cv2.VideoCapture('comp23_2.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.25)

        masks = results[0].masks.xy

        grid = np.zeros((frame.shape[0], frame.shape[1]))

        for mask in masks:
            mask = np.array(mask, dtype=np.int32)
            instance_mask = np.ones((frame.shape[0], frame.shape[1]))
            cv2.fillPoly(instance_mask, [mask], 0)
            grid = np.logical_or(grid, instance_mask)

        occupancy_grid_display = grid.astype(np.uint8) * 255
        #overlay = cv2.addWeighted(frame, 1, cv2.cvtColor(occupancy_grid_display, cv2.COLOR_GRAY2BGR), 0.5, 0)

        transformed_image = getBirdView(occupancy_grid_display, ZED)
        
        current_pixel_size = 0.006  # current size each pixel represents in meters
        desired_pixel_size = 0.05  # desired size each pixel should represent in meters
        scale_factor = current_pixel_size / desired_pixel_size

        # Resize the transformed image
        new_size = (int(transformed_image.shape[1] * scale_factor), int(transformed_image.shape[0] * scale_factor))
        resized_image = cv2.resize(transformed_image, new_size, interpolation = cv2.INTER_AREA)
        print(new_size)
        #append robot location 
        rob_arr = np.zeros((20,161) , dtype=np.uint8)
        

        combined_arr = np.concatenate((resized_image, rob_arr), axis=0)
        rob_arr = cv2.cvtColor(rob_arr, cv2.COLOR_GRAY2BGR)
        combined_arr = cv2.cvtColor(combined_arr, cv2.COLOR_GRAY2BGR)
        cv2.circle(combined_arr, (80, 77), 5, (0, 0, 255), -1) ## just for show 
        # cv2.imshow('rob_array', rob_arr)
        cv2.imshow('image', combined_arr)
        cv2.imshow('Occupancy Grid', transformed_image)
        # numpy_to_occupancy_grid(arr, info=None):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




    # cv2.namedWindow('image')
    # cv2.setMouseCallback('image', click_event, transformed_image)

    # while True:
    #     cv2.imshow('image', transformed_image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()

    # pixel_distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    # #print out image dimensions
    # print(transformed_image.shape)

    # # Assume the real-world distance between the two points is 1 foot (0.3048 meters)
    # real_world_distance = 0.3048  # meters

    # # Calculate the meters per pixel ratio
    # meters_per_pixel = real_world_distance / pixel_distance
    # print(f"Each pixel represents {meters_per_pixel} meters.")

    #cv2.imwrite('test-new-out.jpg', transformed_image)

    # height, width, _ = transformed_image.shape
    # print(f"The image has {height * width} pixels.")

    # point1 = (x1, y1)
    # point2 = (x2, y2)

    # pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # # Assume the real-world distance between the two points is 1 foot (0.3048 meters)
    # real_world_distance = 0.3048  # meters

    # # Calculate the meters per pixel ratio
    # meters_per_pixel = real_world_distance / pixel_distance
    # print(f"Each pixel represents {meters_per_pixel} meters.")


if __name__ == '__main__':
    main()