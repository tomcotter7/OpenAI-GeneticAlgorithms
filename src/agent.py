from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout
import tensorflow as tf


class Agent:

    def __init__(self, name):

        self.nn = self.build_model()
        self.name = name

    def update_name(self, new_name):
      self.name = new_name

    def build_model(self):

        model = tf.keras.Sequential()
        model.add(Conv2D(32, [8, 8], strides=(4, 4),
                  activation="relu", input_shape=(84, 84, 1)))
        model.add(Conv2D(64, [4, 4], strides=(4, 4), activation="relu"))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(6))
        return model

    def update_weights(self, new_weights):
        for index, layer in enumerate(self.nn.layers):
            layer.set_weights(new_weights[index])

    def get_weights(self):
        return [layer.get_weights() for layer in self.nn.layers]
