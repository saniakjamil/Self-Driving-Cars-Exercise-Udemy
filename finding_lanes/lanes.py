import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    # image.shape 0: y max, 1: x max
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def get_average_slope_lines(image, lines):
    left_fit = [] #coordinates of line of best fit on left_fit
    right_fit = [] # for right
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1) # polynomial degree, this is one degreee i.e y = mx + b
        # it will retturn a vector of coefficence which decribes a slope of line
        # print(parameters) This will have array of arrays with 0: slope, 1: y-intercept
        slope = parameters[0]
        intercept = parameters[1]
        # print(slope)
        # left lines will have slope<0 and on right will have >0, now remember
        # slope > 0 if x increases and y increases, else it is not the case
        # in our plot, y increases downwards
        if slope < 0:
            # print(slope, intercept)
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    # print(left_fit)
    # print(right_fit)
    # Now we want average slope for both, so one by one
    left_fit_avg = np.average(left_fit, axis=0) # axis=0, avg them out vertically.
    right_fit_avg = np.average(right_fit, axis=0)
    # print('left: ', left_fit_avg, ' right: ', right_fit_avg)
    # We want to get points of these best_fit lines now
    left_line = get_coordinates(image, left_fit_avg)
    right_line = get_coordinates(image, right_fit_avg)
    return np.array([left_line, right_line])

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    # between 50 an 150 are accepted
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # x1, y1, x2, y2 = line.reshape(4) : no need to reshape since it is alreaddy that way.
            # and we can destructure in loop, it is better
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
average_lines = get_average_slope_lines(lane_image, lines)
#  cv2.HoughLinesP(image, houghSpace grid with 2 px, with 1degree, threshold = 100 intersection atleast,
 # empty array to store these lines, min length off line, max gap between a line)
line_image = display_lines(lane_image, average_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.addWeightes(first image, multiply weight of 0.8, line_image, multiply weight of 1, gamma argument scale 1: add to our sum)

cv2.imshow("region of interest", combo_image)
cv2.waitKey()
# plt.imshow(combo_image)
# plt.show()
