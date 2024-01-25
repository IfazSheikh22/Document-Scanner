# Import the necessary libraries
import cv2  # OpenCV library for computer vision
import numpy as np  # NumPy library for numerical operations

# Load the image
image = cv2.imread('document.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur the image to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges in the image
edges = cv2.Canny(blurred, 75, 200)

# Find contours in the edged image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

# Loop over the contours
for contour in contours:
    # Approximate the contour
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    # If our approximated contour has four points, then we can assume that we have found our document
    if len(approx) == 4:
        documentContour = approx
        break

# Apply a perspective transform to obtain a top-down view of the document
def transform(image, contour):
    # Obtain a consistent order of the points and unpack them individually
    rect = np.zeros((4, 2), dtype = "float32")
    s = contour.sum(axis = 1)
    rect[0] = contour[np.argmin(s)]
    rect[2] = contour[np.argmax(s)]
    diff = np.diff(contour, axis = 1)
    rect[1] = contour[np.argmin(diff)]
    rect[3] = contour[np.argmax(diff)]
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Construct the set of destination points to obtain a "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the warped image
    return warped

warped = transform(image, documentContour.reshape(4, 2))

# Show the original and scanned images
cv2.imshow("Original", image)
cv2.imshow("Scanned", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

""" Simple Document Scanner
This is a simple document scanner implemented in Python using OpenCV. The program takes an image of a document as input, detects the edges of the document, and applies a perspective transform to get a top-down view of the document.

To run the program, you need to have Python and OpenCV installed on your machine. You also need an image of a document named ‘document.jpg’ in the same directory as the script. You can then run the program with the following command: python document_scanner.py
"""