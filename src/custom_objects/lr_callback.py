import tensorflow as tf
import matplotlib.pyplot as plt


class LearningRateLossSave(tf.keras.callbacks.Callback):
    """ A callback for finding learning rate

        Args:
            q (float): value to multiply learning rate by on batch end.

        Methods:
            on_batch_end: saves loss and learning rate on batch end,
                          stops training when loss is greater than 5.

            make_chart: visualizes data gathered by on_batch_end.
    """

    def __init__(self, q: float):
        super().__init__()
        self.q = q
        self.losses = []
        self.rates = []

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs['loss'])
        self.rates.append(self.model.optimizer.learning_rate.read_value())

        if logs['loss'] > 1:
            self.model.stop_training = True

        tf.keras.backend.set_value(self.model.optimizer.learning_rate,
                                   self.model.optimizer.learning_rate.read_value() * self.q)

    def make_chart(self):
        plt.plot(self.rates, self.losses)
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.savefig('lr_to_loss_chart.png')
        plt.show()
