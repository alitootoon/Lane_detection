import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 10)

def average_slope_intercept(lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:  # Left lane
                left_fit.append((slope, y1 - slope * x1))
            else:  # Right lane
                right_fit.append((slope, y1 - slope * x1))

    left_line = np.mean(left_fit, axis=0) if left_fit else None
    right_line = np.mean(right_fit, axis=0) if right_fit else None

    return left_line, right_line

def make_coordinates(img, line_params):
    if line_params is None:
        return None
    slope, intercept = line_params
    y1 = img.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blur, 50, 150)

    # Define the region of interest
    height, width = frame.shape[:2]
    roi_vertices = [
        (0, height),
        (width // 2, int(height * 0.6)),
        (width, height)
    ]
    cropped_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=150
    )

    # Average and extrapolate lines
    left_line, right_line = average_slope_intercept(lines)
    left_coords = make_coordinates(frame, left_line)
    right_coords = make_coordinates(frame, right_line)

    # Draw lines
    line_image = np.zeros_like(frame)
    if left_coords is not None:
        draw_lines(line_image, [left_coords])
    if right_coords is not None:
        draw_lines(line_image, [right_coords])

    # Overlay the lines on the original frame
    combined = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combined

# Main function to process video
video_path = 'test_video.mp4'  # Replace with the path to your test video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame for lane detection
    processed_frame = process_frame(frame)

    # Show the processed frame
    cv2.imshow('Lane Detection', processed_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()