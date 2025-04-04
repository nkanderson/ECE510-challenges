# convolutional_network.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_cnn(input_shape, num_classes, filters=32, kernel_size=(3, 3), pool_size=(2, 2), dense_units=128):
    """
    Creates a CNN with configurable parameters.

    Args:
        input_shape:  Shape of the input images (e.g., (28, 28, 1) for grayscale 28x28).
        num_classes: Number of output classes.
        filters:      Number of filters in the convolutional layer.
        kernel_size:  Size of the convolutional kernel.
        pool_size:    Size of the max pooling window.
        dense_units:  Number of neurons in the dense layer.
    """
    model = keras.Sequential([
        layers.Conv2D(filters, kernel_size, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size),
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate_cnn(model, x_train, y_train, x_test, y_test, epochs=5, batch_size=32):
    """
    Trains and evaluates the CNN.

    Args:
        model:      The Keras model to train.
        x_train:    Training data.
        y_train:    Training labels.
        x_test:     Test data.
        y_test:     Test labels.
        epochs:     Number of training epochs.
        batch_size: Batch size for training.
    """
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy
