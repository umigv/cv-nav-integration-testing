import cv2
import numpy as np
from math import radians, cos, floor
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
    rows, columns = image.shape[:2]
    print(rows, columns)
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

def apply_custom_color_map(image):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    lut[130, 0, :] = [255, 0, 0]  # Red for 2 (now 131)
    lut[127, 0, :] = [0, 0, 255]  # Blue for -1 (now 130)
    lut[129, 0, :] = [0, 255, 0]  # Black for 1 (now 129)
    lut[128, 0, :] = [255, 255, 255]  # White for 0 (now 128)

    # Offset the image values to make them all positive
    image = image + 128

    # Ensure the image is in the correct format
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Map the image values to the custom color map
    image_mapped = cv2.LUT(image, lut)

    return image_mapped

def main():

    ZED = CameraProperties(54, 68.0, 101.0, 68.0)

    cap = cv2.VideoCapture('comp23_2.mp4')

    while True:
        # Read in an image:
        ret, frame = cap.read()
        if not ret:
            break
        # results = model.predict( frame, conf=0.25)
        results = model.predict(frame, conf=0.25, classes=0)

        masks = results[0].masks.xy

        grid = np.zeros((frame.shape[0], frame.shape[1]))

        for mask in masks:
            mask = np.array(mask, dtype=np.int32)

            instance_mask = np.ones((frame.shape[0], frame.shape[1]))
            cv2.fillPoly(instance_mask, [mask], 0)
            grid = np.logical_or(grid, instance_mask)

        occupancy_grid_display = grid.astype(np.int8) * 255
        #overlay = cv2.addWeighted(frame, 1, cv2.cvtColor(occupancy_grid_display, cv2.COLOR_GRAY2BGR), 0.5, 0)

        transformed_image, bottomLeft, bottomRight, topRight, topLeft, maxWidth, maxHeight  = getBirdView(occupancy_grid_display, ZED)

        

        cv2.imshow('Occ', transformed_image)

        maxHeight = int(maxHeight)
        maxWidth = int(maxWidth)

        mask = np.full((maxHeight, maxWidth), -1, dtype=np.int8)
        pts =  np.array([bottomLeft, [bottomRight[0] - 27, bottomRight[1]], [topRight[0] - 65, topRight[1]], topLeft])
        pts = pts.astype(np.int32)  # convert points to int32
        pts = pts.reshape((-1, 1, 2))  # reshape points
        cv2.fillPoly(mask, [pts], True, 0)

        indicies = np.where(mask == -1)
        transformed_image[indicies] = -1

        add_neg = np.full((transformed_image.shape[0], 66), -1, dtype=np.int8)

        transformed_image = np.concatenate((add_neg, transformed_image), axis=1)
        

        transformed_image = np.where(transformed_image==255, 1, transformed_image)
        transformed_image = np.where((transformed_image != 0) & (transformed_image != 1) & (transformed_image != -1), -1, transformed_image)
        print(bottomLeft, bottomRight)


        np.savetxt('mask.txt', mask, fmt='%d')
        np.savetxt('transformed_image.txt', transformed_image, fmt='%d')

        transformed_color = apply_custom_color_map(transformed_image)
        cv2.imshow('Occupancy', transformed_color)

        
        current_pixel_size = 0.006  # current size each pixel represents in meters
        desired_pixel_size = 0.05  # desired size each pixel should represent in meters
        scale_factor = current_pixel_size / desired_pixel_size

        # Resize the transformed image
        new_size = (int(transformed_image.shape[1] * scale_factor), int(transformed_image.shape[0] * scale_factor))
        resized_image = cv2.resize(transformed_image, new_size, interpolation = cv2.INTER_NEAREST_EXACT)
        print(new_size)

        np.savetxt('resized_image.txt', resized_image, fmt='%d')


        rob_arr = np.full((22, 169), -1, dtype=np.int8)
        rob_arr[10][85] = 2


        # combined_arr = np.concatenate((resized_image, rob_arr), axis=0)
        combined_arr = np.vstack((resized_image, rob_arr))

        # combined_arr = combined_arr.astype(np.int8)
        np.savetxt('combined_arr.txt', combined_arr, fmt='%d')

        unique_values = np.unique(combined_arr)
        print(unique_values)

        combined_arr_color = apply_custom_color_map(combined_arr)



        cv2.imshow('Occupancy Grid', combined_arr_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    #out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()