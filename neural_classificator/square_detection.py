import cv2
import numpy as np


def find_square_contours(image, min_area=6000, max_area=10000):
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    square_contours = []
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
                square_contours.append((x, y, w, h))

    # Sort the squares first by y-coordinate (top to bottom), then by x-coordinate (left to right)
    square_contours.sort(key=lambda r: (r[1], r[0]))
    return square_contours


def process_frame(frame):
    # Add a black border to the frame and convert to grayscale
    frame_with_border = cv2.copyMakeBorder(frame, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    gray = cv2.cvtColor(frame_with_border, cv2.COLOR_BGR2GRAY)

    # Apply gradient morphology to highlight edges
    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(gradient, 80, 255, cv2.THRESH_BINARY)

    return find_square_contours(thresh)


if __name__ == '__main__':
    gif_path = '../data/videos/board (2).gif'
    cap = cv2.VideoCapture(gif_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process each frame except the last one
    for frame_idx in range(total_frames - 1):
        ret, frame = cap.read()
        if not ret:
            break

        squares = process_frame(frame)
        print(f'Frame {frame_idx}: {len(squares)} squares detected')

    cap.release()
