import cv2
import numpy as np

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        points.append([x, y])

# Initialize the list of points
points = []

# Load the image, clone it, and setup the mouse callback function
img = cv2.imread('image.jpg')
clone = img.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_circle)

while True:
    # Display the image and wait for a keypress
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF

    # If the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        img = clone.copy()
        points = []

    # If the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

if len(points) > 0:
    # Calculate the rotated bounding rectangle
    rect = cv2.minAreaRect(np.array(points))
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Draw the rotated rectangle
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

# Display the final image
cv2.imshow("image", img)
cv2.waitKey(0)
