import tensorflow as tf
import matplotlib.pyplot as plt


class LearningRateLossSave(tf.keras.callbacks.Callback):
    """ A callback for finding learning rate

    Args:
        q (float): value to multiply learning rate by on batch end.

    Methods:
        on_batch_end(batch, logs=None):
            Saves loss and learning rate on batch end,
            stops training when loss is greater than 5.

        make_chart():
            Visualizes data gathered by on_batch_end.
    """

    def __init__(self, q: float):
        """
            Args:
                q (float): Value to multiply learning rate by on batch end.
        """
        super().__init__()
        self.q = q
        self.losses = []
        self.rates = []

    def on_batch_end(self, batch, logs=None):
        """
        Callback method called on batch end.

        Saves the loss and learning rate on batch end.
        Stops training when loss is greater than 5.

        Args:
            batch (int): The batch index.
            logs (dict): Dictionary containing the training metrics.
        """
        self.losses.append(logs['loss'])
        self.rates.append(self.model.optimizer.learning_rate.read_value())

        if logs['loss'] > 1:
            self.model.stop_training = True

        tf.keras.backend.set_value(self.model.optimizer.learning_rate,
                                   self.model.optimizer.learning_rate.read_value() * self.q)

    def make_chart(self):
        """
        Visualizes the data gathered by on_batch_end.

        Plots a chart of learning rates against losses and saves it as 'lr_to_loss_chart.png'.
        """
        plt.plot(self.rates, self.losses)
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.savefig('lr_to_loss_chart.png')
        plt.show()
