import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("MobileNetV2_CNN_model.h5")

# Classes
CLASSES_LIST = ["non_violence", "violence"]

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is default webcam

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

frame_count = 0  # Added frame counter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 5 != 0:  # process every 5th frame
        continue

    # Preprocess the frame
    img = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    img = img / 960.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    pred = model.predict(img)
    class_index = np.argmax(pred)
    confidence = pred[0][class_index]

    # Display prediction on the frame
    label = f"{CLASSES_LIST[class_index]} ({confidence*100:.2f}%)"
    color = (0, 255, 0) if class_index == 0 else (0, 0, 255)  # Green=non-violence, Red=violence
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the frame
    cv2.imshow("Real-Time Violence Detection", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
