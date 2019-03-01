import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    # between 50 an 150 are accepted
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # destructures 2 points in 4 corrdinates two x and two y
            cv2.line(line_image, (x1,y1), (x2, y2), (255, 0, 0), 10)
            # cv2.line(image, point1, point2, color, width) - draws on line_image these lines
    return line_image

def reigon_of_interst(image):
    height = image.shape[0]
    triangle = np.array([
    [(200, height), (1050, height), (550, 250)]
    ])
    # array of polygons
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = reigon_of_interst(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#  cv2.HoughLinesP(image, houghSpace grid with 2 px, with 1degree, threshold = 100 intersection atleast,
 # empty array to store these lines, min length off line, max gap between a line)
line_image = display_lines(lane_image, lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.addWeightes(first image, multiply weight of 0.8, line_image, multiply weight of 1, gamma argument scale 1: add to our sum)

cv2.imshow("region of interest", combo_image)
cv2.waitKey()
# plt.imshow(combo_image)
# plt.show()
