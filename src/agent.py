import tensorflow as tf


class Agent:

    def __init__(self):

        self.nn = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            64, input_shape=(61,), activation="relu"))
        model.add(tf.keras.layers.Dense(32, activation="relu"))
        model.add(tf.keras.layers.Dense(20, activation="sigmoid"))
        return model

    def update_weights(self, new_weights):
        for index, layer in enumerate(self.nn.layers):
            layer.set_weights(new_weights[index])

    def get_weights(self):
        return [layer.get_weights() for layer in self.nn.layers]
