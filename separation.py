import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the input image
img = cv2.imread('hnst2.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to create a binary mask
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)

# Apply closing to remove small noise and connect nearby foreground pixels
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours in the binary mask
contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area
max_area = 0
max_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

mask = np.zeros_like(thresh)
cv2.drawContours(mask, [max_contour], 0, 255, -1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Apply a Gaussian blur to smooth the edges of the mask
blur = cv2.GaussianBlur(opening, (51, 51), 0)

# Convert the mask to three channels to use for merging
mask = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

# Invert the mask to get the background
background = cv2.bitwise_and(img, cv2.bitwise_not(mask))

# Get the foreground by applying the mask to the input image
foreground = cv2.bitwise_and(img, mask)

# Blend the foreground and background using bitwise OR
merged = cv2.bitwise_or(foreground, background)

# Display the original and merged images
cv2_imshow(img)
cv2_imshow(background)
cv2.imwrite('background.jpg',background)
cv2_imshow(foreground)
cv2.imwrite('foreground.jpg',foreground)
cv2_imshow(merged)

cv2.waitKey(0)
cv2.destroyAllWindows()
