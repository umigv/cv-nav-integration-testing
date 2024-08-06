import cv2  
import numpy as np
from math import radians, cos, sqrt


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
    #print(rows, columns)
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
    print("Matrix")
    print(image.shape)
    warped = cv2.warpPerspective(image, matrix1, (cp.maxWidth, cp.maxHeight))


    return warped, bottomLeft, bottomRight, topRight, topLeft

points = []

img = cv2.imread("bird2.png")

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at the clicked point
        if len(points) == 2:
            distance = sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
            print(f"Distance: {distance}")
            cv2.line(img, points[0], points[1], (255, 0, 0), 2)  # Draw a blue line between the points
            cv2.putText(img, f"{distance:.2f}", (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.imshow("image with distance", img)  # Display the image with the distance in a new window
            points.clear()

def main():
    # img = cv2.imread("test.png")

    # split = img.shape[1]//2
    # left = img[:, :split]

    # ZED = CameraProperties(64, 68.0, 101.0, 60.0)
    # # ZED = CameraProperties(54, 68.0, 101.0, 68.0)

    # bird, bottomLeft, bottomRight, topRight, topLeft, maxWidth, maxHeight = getBirdView(left, ZED)

    # # # save the bird's eye view
    # cv2.imwrite("bird2.png", bird)

    cv2.namedWindow("image")



    cv2.setMouseCallback("image", click_event)

    cv2.imshow("image", img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()



    # cv2.waitKey(0)

if __name__ == "__main__":
    main()

