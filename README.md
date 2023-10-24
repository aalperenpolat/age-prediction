# Age Estimation and Face Recognition Project

This project captures images from a camera and performs face detection to estimate the age of detected faces. Face recognition is achieved using OpenCV, and age estimation is done using Tensorflow and Keras.

## Requirements

To run this project, you need the following requirements:

- Python 3.x
- OpenCV
- Tensorflow
- Keras
- NumPy

You can install the requirements using pip. For example:

```bash
pip install opencv-python
pip install tensorflow
pip install numpy

```



Usage
When the project is run, it captures images from the connected camera. It detects faces in the captured image and adds an estimated age to each detected face. While the project is running, the estimated age of each face is displayed on the output screen.

To close the output screen, you can press the "q" key.

Model Training
The age estimation model loads a pre-trained model. If you wish to train your own model with your own data, you can find more information and training steps in the model_training folder.

Contributions
If you would like to contribute to the project, please refer to the CONTRIBUTING.md file for guidelines.

License
This project is distributed under the MIT license. For more information, refer to the LICENSE file.
