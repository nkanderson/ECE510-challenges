import unittest
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import logging
import os

def create_cnn(input_shape, num_classes):
    """Creates a convolutional neural network model."""
    model = keras.Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate_cnn(model, x_train, y_train, x_test, y_test, epochs=5):
    """Trains and evaluates a CNN model."""
    model.fit(x_train, y_train, epochs=epochs, verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy

class TestCNN(unittest.TestCase):
    def test_cnn_training(self):
        """Tests CNN training on a small, random dataset."""
        input_shape = (28, 28, 1)
        num_classes = 10
        model = create_cnn(input_shape, num_classes)
        x_train = np.random.rand(100, 28, 28, 1).astype(np.float32)
        y_train = np.random.randint(0, 10, 100).astype(np.int32)
        x_test = np.random.rand(20, 28, 28, 1).astype(np.float32)
        y_test = np.random.randint(0, 10, 20).astype(np.int32)
        loss, accuracy = train_and_evaluate_cnn(model, x_train, y_train, x_test, y_test, epochs=3)
        self.assertGreaterEqual(accuracy, 0.1)
        self.assertGreaterEqual(loss, 0)



def main():
    """Main function to run the CNN program."""
    print("Running CNN program execution...")
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all, 1 = info, 2 = warning, 3 = error
    logging.getLogger('tensorflow').setLevel(logging.ERROR)


    # Example
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    model = create_cnn((28, 28, 1), 10)
    loss, accuracy = train_and_evaluate_cnn(model, x_train, y_train, x_test, y_test, epochs=5)
    print(f"CNN - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    unittest.main()
