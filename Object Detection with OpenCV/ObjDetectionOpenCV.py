import cv2
import numpy as np
from collections import defaultdict

# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()

# Get output layers
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the camera
cap = cv2.VideoCapture(0)  # Change the index if needed

def detect_and_count_objects(frame):
    height, width, _ = frame.shape

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Run the forward pass
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Process the outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Count objects
    object_count = defaultdict(int)
    if len(indexes) > 0:
        for i in indexes[0]:  # Access the first element of the tuple
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            object_count[label] += 1  # Increment the count for the detected object
            
            # Draw box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return object_count

while True:
    ret, frame = cap.read()
    if not ret:
        break

    object_count = detect_and_count_objects(frame)

    # Display counts on the frame
    count_text = "Counts: " + ", ".join([f"{k}: {v}" for k, v in object_count.items()])
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Object Detection and Counting', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()