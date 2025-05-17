from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Load and preprocess dataset
data = []
labels = []
cur_path = os.getcwd()

classes = {
    0:"Speed limit (20km/h)", 1:"Speed limit (30km/h)", 2:"Speed limit (50km/h)", 3:"Speed limit (60km/h)",
    4:"Speed limit (70km/h)", 5:"Speed limit (80km/h)", 6:"End of speed limit (80km/h)", 7:"Speed limit (100km/h)",
    8:"Speed limit (120km/h)", 9:"No overtaking", 10:"No overtaking with a heavy trucks", 11:"Right of way",
    12:"Priority road", 13:"Yield", 14:"Stop", 15:"No vehicles", 16:"Goods vehicle exceeding 3.5T prohibited", 17:"No Entry",
    18:"General cautions", 19:"Dangerous curve to the left", 20:"Dangerous curve to the right", 21:"Double curve",
    22:"Bumpy road", 23:"Slippery road", 24:"Road narrows on the right", 25:"Road work ahead", 26:"Traffic light signals",
    27:"Pedestrian crossing", 28:"Children crossing", 29:"Bicycles Crossing", 30:"Beware of ice/snow", 31:"Wild animals",
    32:"End Speed & passing limit", 33:"Turn right ahead", 34:"Turn left ahead", 35:"Ahead only",
    36:"Go straight or turn right", 37:"Go straight or turn left", 38:"Keep right", 39:"Keep left",
    40:"Round about mandatory", 41:"End of no overtaking", 42:"End of no overtaking for trucks"
}

print("Loading dataset...")
for i in range(43):
    path = os.path.join(cur_path, 'dataset/train', str(i))
    if not os.path.exists(path):
        print(f"Folder not found: {path}")
        continue
    for a in os.listdir(path):
        print(f"Loading: {a}")
        try:
            img = Image.open(os.path.join(path, a))
            img = img.resize((30, 30))
            data.append(np.array(img))
            labels.append(i)
        except:
            print(f"Error loading image: {a}")

data = np.array(data)
labels = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(550, 550)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.headingLabel = QtWidgets.QLabel(self.centralwidget)
        self.headingLabel.setGeometry(QtCore.QRect(30,10,500,90))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(True)
        self.headingLabel.setFont(font)
        self.headingLabel.setText("ROAD SIGN RECOGNITION")
        self.headingLabel.setAlignment(QtCore.Qt.AlignCenter)
        

        self.Browsebutton = QtWidgets.QPushButton("Browse", self.centralwidget)
        self.Browsebutton.setGeometry(QtCore.QRect(40, 350, 141, 71))

        self.Classifybutton = QtWidgets.QPushButton("Classify", self.centralwidget)
        self.Classifybutton.setGeometry(QtCore.QRect(40, 460, 141, 71))

        self.Trainingbutton = QtWidgets.QPushButton("Train", self.centralwidget)
        self.Trainingbutton.setGeometry(QtCore.QRect(350, 460, 131, 71))

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(130, 90, 291, 211))
        self.label.setStyleSheet("border: 1px solid black")
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(350, 350, 131, 71))

        MainWindow.setCentralWidget(self.centralwidget)

        # Connect buttons
        self.Browsebutton.clicked.connect(self.loadImage)
        self.Classifybutton.clicked.connect(self.ClassifyFunction)
        self.Trainingbutton.clicked.connect(self.trainingFunction)

    def loadImage(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Images (*.png *.jpg *.bmp)")
        if filename:
            self.file = filename
            pixmap = QtGui.QPixmap(filename)
            pixmap = pixmap.scaled(self.label.width(), self.label.height(), QtCore.Qt.KeepAspectRatio)
            self.label.setPixmap(pixmap)
            self.label.setAlignment(QtCore.Qt.AlignCenter)

    def ClassifyFunction(self):
        model = load_model('my_model.h5')
        test_image = Image.open(self.file)
        test_image = test_image.resize((30, 30))
        test_image = np.expand_dims(test_image, axis=0)
        test_image = np.array(test_image)

        result = np.argmax(model.predict(test_image), axis=1)[0]
        sign = classes[result]
        self.textEdit.setText(sign)

    def trainingFunction(self):
        self.textEdit.setText("Training...")
        model = Sequential()
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=x_train.shape[1:]))
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(43, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))
        model.save("my_model.h5")

        plt.figure(0)
        plt.plot(history.history['accuracy'], label='training accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.legend()
        plt.savefig("Accuracy.png")

        plt.figure(1)
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.legend()
        plt.savefig("Loss.png")

        self.textEdit.setText("Model trained and saved.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
