import cv2 as cv
import numpy as np

def get_contour_center(contour):
    M = cv.moments(contour)
    cx=-1
    cy=-1
    if (M['m00']!=0):
        cx= int(M['m10']/M['m00'])
        cy= int(M['m01']/M['m00'])
    return cx, cy

def draw_circle(rgb_image, contours) :
    for c in contours :
        cx, cy = get_contour_center(c)
        ((x, y), radius) = cv.minEnclosingCircle(c)
        cv.circle(rgb_image, (cx,cy),(int)(radius),(0,0,255),4)

    return rgb_image

video_capture = cv.VideoCapture("Videos/tennis-ball-video.mp4")

# Empty function for createTrackbar command
def empty(a) :
    pass

# Creating Trackbars
cv.namedWindow("Trackbar")
cv.createTrackbar("Hue Min", "Trackbar", 30, 179,  empty)
cv.createTrackbar("Hue Max", "Trackbar", 60, 179,  empty)
cv.createTrackbar("Saturation Min", "Trackbar", 100, 255,  empty)
cv.createTrackbar("Saturation Max", "Trackbar", 255, 255,  empty)
cv.createTrackbar("Value Min", "Trackbar", 100, 255,  empty)
cv.createTrackbar("Value Max", "Trackbar", 255, 255,  empty)

window_width = 500
window_height = 300

cv.namedWindow("Video", cv.WINDOW_NORMAL)
cv.resizeWindow("Video", window_width, window_height)

cv.namedWindow("Binary Mask", cv.WINDOW_NORMAL)
cv.resizeWindow("Binary Mask", window_width, window_height)

cv.namedWindow("Circled Video", cv.WINDOW_NORMAL)
cv.resizeWindow("Circled Video", window_width, window_height)



while True :
    net, frame = video_capture.read()
    image = frame

    aspect_ratio = frame.shape[1] / frame.shape[0]
    new_height = int(window_width / aspect_ratio)
    frame = cv.resize(frame, (window_width, new_height))

    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    min_hue = cv.getTrackbarPos("Hue Min", "Trackbar")
    max_hue = cv.getTrackbarPos("Hue Max", "Trackbar")
    min_sat = cv.getTrackbarPos("Saturation Min", "Trackbar")
    max_sat = cv.getTrackbarPos("Saturation Max", "Trackbar")
    min_val = cv.getTrackbarPos("Value Min", "Trackbar")
    max_val = cv.getTrackbarPos("Value Max", "Trackbar")

    print(min_hue, min_sat, min_val, max_hue, max_sat, max_val)

    lower_bound = np.array([min_hue, min_sat, min_val])
    upper_bound = np.array([max_hue, max_sat, max_val])

    binary_mask = cv.inRange(image_hsv, lower_bound, upper_bound)
    contours, hierarchy = cv.findContours(binary_mask.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    new_image = image.copy()
    new_image = draw_circle(new_image, contours)

    cv.imshow("Video", image)
    cv.imshow("Binary Mask", binary_mask)
    cv.imshow("Circled Video", new_image)
    cv.waitKey(1)

cv.destroyAllWindows()