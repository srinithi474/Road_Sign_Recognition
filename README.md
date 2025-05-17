
# Road Sign Recognition (GUI with Keras & PyQt5)

This is a Python project that uses a simple graphical interface (GUI) built with PyQt5 to classify road signs using a Convolutional Neural Network (CNN). The user can:

- Load a traffic sign image
- Train the CNN model (using GTSRB dataset)
- Click “Classify” to display the predicted sign name in a text box

It's an educational project to demonstrate how deep learning models can be integrated with a desktop user interface.

## Features

- Load and display any traffic sign image from your device
- Train a model with one click inside the GUI
- Classify the loaded image using a trained model
- View prediction results in a simple text box
- Visualize accuracy/loss curves (saved after training)

## Technologies Used

- Python
- PyQt5
- Keras / TensorFlow
- NumPy
- PIL
- Matplotlib
- scikit-learn

## How to Run

1. Clone the repo or download the files
2. Install dependencies:
   ```bash
   pip install PyQt5 tensorflow numpy matplotlib pillow scikit-learn
   ```
3. Run the script:
   ```bash
   python main.py
   ```
4. Use the buttons:
   - **Browse**: Load a traffic sign image
   - **Train**: Train the CNN model on GTSRB data
   - **Classify**: Predict the traffic sign and display the result

## Dataset

Model is trained on the [GTSRB dataset](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset), which contains 43 classes of German traffic signs.

## Note

- This is not a standalone app installer. It runs as a Python script and displays a simple window using PyQt5.

## Author

Created by Srinithi as a part of my learning journey in deep learning and GUI development.
