import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
tflite_model_path = "/Users/mathew/Desktop/git py projects/midas.tflite"  # Replace with your TFLite model path
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Webcam setup
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input frame
    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]
    resized_frame = cv2.resize(frame, (input_width, input_height))
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    depth_map = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]

    # Normalize depth map for visualization
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Generate heatmap
    heatmap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # Display original frame and heatmap
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Depth Heatmap', heatmap)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
