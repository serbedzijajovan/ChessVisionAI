import cv2
import numpy as np


def find_square_contours(img, image_bin, min_area=6000, max_area=10000):
    contours, _ = cv2.findContours(image_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    square_contours = []
    rectangles = []
    for contour in contours:
        # Approximate the contour to check if it's a square
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:  # Check if contour has 4 sides (square)
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(approx)

            # Check if the contour is a square based on aspect ratio and area
            if 0.9 <= aspect_ratio <= 1.1 and min_area <= area <= max_area:
                square = img[y:y + h + 1, x:x + w + 1]
                resized_square = cv2.resize(square, (90, 90), interpolation=cv2.INTER_NEAREST)
                square_contours.append(resized_square)
                rectangles.append((x, y, w, h))

    sorted_tuples = sorted(zip(rectangles, square_contours), key=lambda b: (b[0][1], b[0][0]))
    _, sorted_square_contours = zip(*sorted_tuples)
    return sorted_square_contours


def process_frame(frame):
    """Processes the frame in RGB to extract square regions."""
    # Add a black border to the frame and convert to grayscale
    frame_with_border = cv2.copyMakeBorder(frame, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    frame_gray = cv2.cvtColor(frame_with_border, cv2.COLOR_RGB2GRAY)

    # Apply gradient morphology to highlight edges
    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(frame_gray, cv2.MORPH_GRADIENT, kernel)

    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(gradient, 70, 255, cv2.THRESH_BINARY_INV)

    return find_square_contours(frame, thresh)
