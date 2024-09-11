import cv2
from pyzbar.pyzbar import decode

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution (optional for performance)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set up a set to keep track of decoded codes
decoded_codes = set()

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Decode barcodes/QR codes
    codes = decode(gray)

    for code in codes:
        decoded_data = code.data.decode('utf-8')

        if decoded_data not in decoded_codes:
            print(f"New code detected: {decoded_data}")
            decoded_codes.add(decoded_data)

            # Draw green rectangle for new codes
            (x, y, w, h) = code.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, decoded_data, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Draw blue rectangle for already detected codes
            (x, y, w, h) = code.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, decoded_data, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()