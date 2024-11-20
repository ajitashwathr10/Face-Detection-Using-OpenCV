# Face Detection Using OpenCV

This repository contains a simple implementation of face detection using OpenCV and Python. The project uses Haar Cascades for detecting faces with the help of a picture.

## Requirements
To run this project, you need to have the following libraries installed:
- Python 3.12
- OpenCV
- NumPy

You can install the required libraries using `pip`:

```bash
pip install opencv-python numpy
```
## Project Structure
```bash
Face-Detection-Using-OpenCV/
│
├── Face_Detection.ipynb         # Main script for face detection
├── face_detection.xml           # Directory containing Haar Cascade XML files
├── face_detected.png    
├── README.md                # Project documentation
└── female.png        
```
## Working
1. Load the Haar Cascade XML file (pre-trained model).
2. Load an image where faces need to be detected.
3. Convert the image to grayscale (as the classifier works better on grayscale images).
4. Detect faces in the image using the `cv2.CascadeClassifier` method.
5. Draw bounding boxes around the detected faces.
6. Save or display the image with the detected faces.

## Code Explanation
In the `face_detection.py` file, the following steps are performed:
1. Loading the Classifier:
   The Haar Cascade XML file is loaded using cv2.CascadeClassifier:
   ```bash
   face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
   ```
2. Capturing Video:
   OpenCV captures video using the cv2.VideoCapture(0) method, where 0 refers to the default webcam.
   ```bash
   cap = cv2.VideoCapture(0)
   ```
3. Face Detection:
   The image is first converted to grayscale using:
   ```bash
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   ```
   Then, faces are detected with:
   ```bash
   faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
   ```
4. Drawing Bounding Boxes:
   Detected faces are highlighted with bounding boxes using:
   ```bash
   for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
   ```
5. Displaying or Saving the Image:
   The result is shown or saved with the bounding boxes drawn around face:
   ```bash
   cv2.imshow('Face Detection', image)
   cv2.imwrite('output_image.jpg', image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/ajitashwathr10/Face-Detection-Using-OpenCV.git
   cd Face-Detection-Using-OpenCV
   ```
2. Place an image file in the project directory (or update the file path in the code).
3. Run the main script:
   ```bash
   python face_detection.py
   ```
4. The image will be processed, and the detected faces will be highlighted with bounding boxes. The resulting image will either be displayed or saved as `output_image.jpg`.

## Acknowledgements
- This project uses the `Haar Cascade Classifier` provided by OpenCV.
- Thanks to the OpenCV community for their valuable contributions.

