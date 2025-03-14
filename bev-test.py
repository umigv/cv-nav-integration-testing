import cv2
import numpy as np
from math import radians, cos

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
            self.bird_src_quad = np.array([[0, rows - 1], [columns - 1, rows - 1], [0, 0], [columns - 1, 0]], dtype='float32')
        return self.bird_src_quad

    def dst_quad(self, rows, columns, min_angle, max_angle):
        if self.bird_dst_quad is None:
            fov_offset = self.cameraTilt - self.fov_vert / 2.0
            bottom_over_top = cos((max_angle * 1.2) + fov_offset) / cos(min_angle + fov_offset)
            bottom_width = columns * bottom_over_top
            blackEdge_width = (columns - bottom_width) / 2
            leftX = blackEdge_width
            rightX = leftX + bottom_width
            self.bird_dst_quad = np.array([[leftX, rows], [rightX, rows], [0, 0], [columns, 0]], dtype='float32')
        return self.bird_dst_quad

    def reset(self):
        self.bird_src_quad = None
        self.bird_dst_quad = None
        self.matrix = None
        self.maxHeight = None
        self.maxWidth = None
        self.minIndex = None

    def compute_min_index(self, rows, max_angle):
        self.minIndex = int(rows * (1.0 - max_angle / self.fov_vert))
        return self.minIndex

    def compute_max_angle(self):
        return min(CameraProperties.functional_limit - self.cameraTilt + self.fov_vert / 2.0, self.fov_vert * 1.5)

def getBirdView(image, cp):
    rows, columns = image.shape[:2]
    min_angle = 0.0
    max_angle = cp.compute_max_angle()
    min_index = cp.compute_min_index(rows, max_angle)
    image = image[min_index:, :]
    rows = image.shape[0]

    src_quad = cp.src_quad(rows, columns)
    dst_quad = cp.dst_quad(rows, columns, min_angle, max_angle)
    warped, bottomLeft, bottomRight, topRight, topLeft = perspective(image, src_quad, dst_quad, cp)
    return warped

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

def main():
    ZED = CameraProperties(64, 68.0, 101.0, 40.0)
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        #get the left half of the image
        frame = frame[:, :frame.shape[1] // 2]
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        bird_view_image = getBirdView(frame, ZED)
        #only keep heights from 305 - bird_view_image.shape[0] (to remove the top of the image)
        bird_view_image = bird_view_image[305:bird_view_image.shape[0], :]
        print(bird_view_image.shape)
        # combined_image = np.vstack((frame, bird_view_image))
        cv2.imshow('BEV', bird_view_image)
        cv2.imshow('Original and Bird\'s Eye View', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()