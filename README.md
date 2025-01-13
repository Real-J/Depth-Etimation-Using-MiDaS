# Depth-Etimation-Using-MiDaS

## Features

- **Depth Estimation**: Leverages the MiDaS v2.1 small model (TFLite) for depth prediction.
- **Real-Time Processing**: Processes frames from a webcam in real time.
- **Heatmap Visualization**: Generates a visually intuitive heatmap of depth data.
- **Cross-Platform Compatibility**: Runs on most systems with Python and TensorFlow Lite installed.

---

## Installation

### Prerequisites
Ensure you have the following installed on your system:

- Python 3.7 or later
- TensorFlow 2.x
- OpenCV
- NumPy

### Install Required Libraries
To install the necessary Python libraries, run:

```bash
pip install tensorflow opencv-python numpy
```

---

## Download the Pre-trained Model

The script uses the MiDaS v2.1 small TFLite model. Follow these steps to download it:

1. Visit [Kaggle for MiDaS TFLite](https://www.kaggle.com/models/intel/midas/tfLite/v2-1-small-lite/1?tfhub-redirect=true).
2. Download the file `midas_v2_1_small_256.tflite`.
3. Save the file in the same directory as the script or specify the path in the code.

---

## Usage

### Clone the Repository

```bash
git clone https://github.com/Real-J/Depth-Estimation-Using-MiDaS.git
cd depth-estimation-tflite
```

### Run the Script

Ensure the TFLite model is in the correct location, then run:

```bash
python depthestimation.py
```

### Keyboard Commands

- Press `q` to quit the application.

---

## Code Overview

### Key Components

1. **Model Loading**:
   - The TensorFlow Lite Interpreter is used to load the MiDaS model.
   - Input and output tensor information is extracted for inference.

2. **Webcam Input**:
   - OpenCV captures video frames in real time.
   - Frames are resized and normalized to match the model's input requirements.

3. **Depth Estimation**:
   - Frames are passed through the TFLite model to generate a depth map.
   - The depth map is normalized to enhance visualization.

4. **Heatmap Generation**:
   - The normalized depth map is converted into a heatmap using OpenCV's `applyColorMap` function.

5. **Display**:
   - Both the original frame and the heatmap are displayed side by side in real time.


---

## Expected Output

When running the script, two windows will appear:

1. **Original Frame**: Displays the live feed from the webcam.
2. **Depth Heatmap**: Visualizes the estimated depth as a heatmap.

The heatmap uses colors to represent depth:

- **Red**: Objects closer to the camera.
- **Blue**: Objects farther from the camera.

---

## Troubleshooting

### Common Issues

1. **Model Not Found**:
   - Ensure the `midas_v2_1_small_256.tflite` file is in the correct directory.
   - Update the `tflite_model_path` variable with the correct path.

2. **Webcam Not Detected**:
   - Verify your webcam is connected.
   - Check that OpenCV is accessing the correct device index (`cv2.VideoCapture(0)`).

3. **Performance Issues**:
   - Ensure your system meets the requirements.
   - Use a smaller input size for faster processing.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [MiDaS Depth Estimation](https://github.com/isl-org/MiDaS)


