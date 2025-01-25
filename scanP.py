import cv2
import numpy as np

# Function to detect and count moving objects crossing a line
def count_moving_objects(video_url="http://100.74.106.90:8080/video", frame_skip=3):
    # Open the IP camera feed (use the URL of your phone camera)
    cap = cv2.VideoCapture(video_url)
    
    # Check if the camera feed is opened correctly
    if not cap.isOpened():
        print("Error: Unable to open video stream")
        return

    # Initialize variables
    object_count = 0
    min_contour_area = 800  # Minimum contour area to consider as an object
    offset = 50  # Offset for detecting the crossing of the line
    previous_centroids = []  # To track the centroids of detected objects
    movement_detected = False  # To flag when motion is detected

    # Define the background subtractor to detect moving objects
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    # Read the first frame to determine frame size
    _, frame = cap.read()
    height, width, _ = frame.shape
    counting_line_y = height // 2  # Middle of the frame for counting

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to reduce processing load
        if frame_skip > 1:
            for _ in range(frame_skip - 1):
                cap.read()

        # Resize the frame to process faster
        frame = cv2.resize(frame, (640, 480))
        
        # Apply background subtraction to isolate moving objects
        fgmask = fgbg.apply(frame)
        
        # Apply morphological transformations to remove noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the counting line
        cv2.line(frame, (0, counting_line_y), (width, counting_line_y), (0, 255, 0), 2)

        # Temporary list to store the centroids of this frame
        current_centroids = []

        # Loop over the contours
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                # Get the bounding box for the contour
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw the bounding box around the object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Calculate the centroid of the object
                centroid_x = x + w // 2
                centroid_y = y + h // 2
                current_centroids.append((centroid_x, centroid_y))
                
                # Check if the centroid crosses the counting line
                if counting_line_y - offset < centroid_y < counting_line_y + offset:
                    # Compare with previous centroids to avoid double counting
                    if all(abs(centroid_x - cx) > 20 for cx, cy in previous_centroids):
                        object_count += 1
                        movement_detected = True  # Motion detected and object counted

        # Display the object count on the frame
        cv2.putText(frame, f"Object Count: {object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Only display the frame when there is movement
        if movement_detected:
            cv2.imshow("Moving Object Counting", frame)
            movement_detected = False  # Reset the movement flag

        # Update previous centroids
        previous_centroids = current_centroids

        # Exit on pressing 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function to start counting with your phone's IP camera
count_moving_objects(video_url="http://100.74.106.90:8080/video", frame_skip=3)
