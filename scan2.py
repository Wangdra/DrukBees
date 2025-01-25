import cv2
from pyzbar.pyzbar import decode
import psycopg2
import time  # For tracking time

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname="druk_bees", 
    user="postgres", 
    password="Wangdrajoker!",
    host="localhost"  # Or the appropriate host if PostgreSQL is hosted elsewhere
)
cur = conn.cursor()

url = "http://100.72.8.4:8080/video"  # Your IP camera or webcam URL
cap = cv2.VideoCapture(url)

# Initialize variables for total, scanned products, their counts, and total product count
total_cost = 0
scanned_products = {}
last_scanned_time = {}  # Dictionary to track the last scanned time for each barcode
frame_count = 0
COOLDOWN_PERIOD = 3  # Cooldown period of 2 seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if there's an issue capturing

    # Resize the frame to higher resolution for better detection of small barcodes
    frame = cv2.resize(frame, (1280, 720))

    # Process only every 10th frame for performance
    if frame_count % 10 == 0:
        # First attempt to decode barcodes on the original frame
        barcodes = decode(frame)

        # If no barcodes detected, try zooming in on the central area
        if not barcodes:
            # Center crop the image to zoom in
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            zoom_size = 200  # Change this value to increase/decrease zoom level
            cropped_frame = frame[center_y - zoom_size:center_y + zoom_size, center_x - zoom_size:center_x + zoom_size]
            cropped_frame = cv2.resize(cropped_frame, (640, 480))  # Resize cropped frame to original size
            barcodes = decode(cropped_frame)

        # Process each detected barcode
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around the barcode
            barcode_data = barcode.data.decode('utf-8')
            barcode_type = barcode.type
            text = f"{barcode_data} ({barcode_type})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Put text on the frame

            print(f"Detected {barcode_type}: {barcode_data}")

            # Get the current time
            current_time = time.time()

            # Check if the barcode has been scanned before
            if barcode_data in last_scanned_time:
                elapsed_time = current_time - last_scanned_time[barcode_data]

                # If the barcode was scanned within the cooldown period, ignore it
                if elapsed_time < COOLDOWN_PERIOD:
                    print(f"Ignoring {barcode_data}. Scanned {elapsed_time:.2f} seconds ago.")
                    continue
                else:
                    print(f"{barcode_data} scanned again after {elapsed_time:.2f} seconds.")
            
            # Update the last scanned time for this barcode
            last_scanned_time[barcode_data] = current_time

            # Fetch product information from the database using the scanned barcode
            cur.execute("SELECT product_name, price FROM products WHERE barcode = %s", (barcode_data,))
            product = cur.fetchone()

            if product:
                product_name, price = product
                print(f"Product Name: {product_name}, Price: {price}")
                
                # Add product to the scanned products list and update the total cost
                if product_name in scanned_products:
                    scanned_products[product_name]['count'] += 1  # Increment the count if already scanned
                else:
                    scanned_products[product_name] = {'price': price, 'count': 1}  # Add new product

                total_cost += price  # Update total cost

                # Display product information on the frame
                product_text = f"Product: {product_name}, Price: {price} Nu."
                cv2.putText(frame, product_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                print("Product not found in database.")

    # Calculate total number of products (sum of all counts)
    total_products_count = sum(info['count'] for info in scanned_products.values())

    # Display the scanned products, their counts, the total cost, and total product count on the frame
    for i, (prod_name, info) in enumerate(scanned_products.items()):
        count = info['count']
        price = info['price']
        cv2.putText(frame, f"{prod_name} - {count} x {price} Nu.", (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.putText(frame, f"Total: {total_cost} Nu.", (10, 100 + len(scanned_products) * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Products: {total_products_count}", (10, 140 + len(scanned_products) * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    frame_count += 1
    cv2.imshow("Barcode/QR Code Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the PostgreSQL connection
cur.close()
conn.close()

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
