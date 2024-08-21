import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


class LearningRateSchedule(tf.keras.callbacks.Callback):
    """ A callback for finding learning rate

    Methods:
        on_batch_end(batch, logs=None):
            Saves loss and learning rate on batch end,
            stops training when loss is greater than 5.

        make_chart():
            Visualizes data gathered by on_batch_end.
    """

    def __init__(self, initial_lr: float, target_lr: float, n_batches: int):
        """
            Args:
                initial_lr (float): Learning rate at the beginning of a training
                target_lr (float): Learning rate we want to hit mid-training
                n_batches (int): Number of batches
        """
        super().__init__()
        self.n_batches = n_batches
        self.n_updates = 0
        self.d = (target_lr - initial_lr) / (n_batches / 2)
        self.rates = []
        self.updates = []

    def on_batch_end(self, batch, logs=None):
        """
        Callback method called on batch end.

        Uses calculated q during init to meet target learning rate value at the middle of training.

        Args:
            batch (int): The batch index.
            logs (dict): Dictionary containing the training metrics.
        """
        if self.n_updates <= self.n_batches / 2:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate,
                                       self.model.optimizer.learning_rate.read_value() + self.d)
        else:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate,
                                       self.model.optimizer.learning_rate.read_value() - self.d)
        self.n_updates += 1
        self.rates.append(self.model.optimizer.learning_rate.read_value())
        self.updates.append(self.n_updates)

    def make_chart(self, fit_time_date: str):
        """
        Visualizes the data gathered by on_batch_end.
        """
        plt.plot(self.updates, self.rates)
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        os.mkdir(os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'lr_schedule',
            fit_time_date
        ))
        plt.savefig(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'lr_schedule',
                fit_time_date,
                'lr_shedule.png'
            )
        )
        plt.show()
