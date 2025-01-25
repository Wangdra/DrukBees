import cv2
import numpy as np

# Load pre-trained SSD model and configuration (Caffe version)
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Open video capture
url = "http://192.168.0.15:8080/video"  # Replace with your camera feed URL
cap = cv2.VideoCapture(url)

# Parameters for counting
line_position = 600  # Adjust the y-coordinate of the bottom line
object_count = 0

# Function to calculate the centroid of a rectangle
def calculate_centroid(x, y, w, h):
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return (cx, cy)

# Function to check if the object has crossed the line
def check_line_crossing(centroid, line_y):
    return centroid[1] > line_y

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video capture fails

    # Resize frame for faster processing (300x300 is SSD input size)
    h, w = frame.shape[:2]
    resized_frame = cv2.resize(frame, (300, 300))

    # Convert the frame to a blob for SSD input
    blob = cv2.dnn.blobFromImage(resized_frame, 0.007843, (300, 300), 127.5)

    # Set the blob as input to the SSD network
    net.setInput(blob)
    detections = net.forward()

    # Draw the bottom line for line crossing detection
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 255), 2)

    # List to store centroids of detected objects
    centroids = []

    # Process detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold (adjust as needed)
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            width = x2 - x
            height = y2 - y

            # Filter for specific size objects (like products)
            if width > 50 and height > 50:  # Adjust for your product size
                # Draw bounding box around the object
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                # Calculate centroid of the object
                centroid = calculate_centroid(x, y, width, height)
                centroids.append(centroid)

                # Draw the centroid of the object
                cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

                # Check if the object crosses the bottom line
                if check_line_crossing(centroid, line_position):
                    object_count += 1
                    # Remove the centroid to avoid multiple counts for the same object
                    centroids.remove(centroid)

    # Display the object count on the frame
    cv2.putText(frame, f"Object Count: {object_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("SSD Object Detection and Counting", frame)

    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
