"""
Evaluate TensorFlow Model on Test Dataset and Plot Confusion Matrix

This script evaluates a TensorFlow model on a test dataset and generates a confusion matrix plot.

Module Dependencies:
    - src.data (imported as test_ds, CSV_DICT, TEST_SET_PATH)
    - tensorflow (imported as tf)
    - src.custom_objects.layer_scale (imported as LayerScale)
    - sklearn.metrics (imported confusion_matrix, ConfusionMatrixDisplay)
    - matplotlib.pyplot (imported as plt)
    - pandas (imported as pd)

Global Constants:
    - CSV_DICT (dict): Dictionary containing paths to CSV files.
    - TEST_SET_PATH (str): Path to the test dataset.
    - model_path (str): Path to the pre-trained TensorFlow model.

Data Loading:
    - Loads the test dataset and ground truth labels from the CSV file.

Model Loading:
    - Loads a pre-trained TensorFlow model from the specified file.

Model Prediction:
    - Uses the loaded model to predict labels for the test dataset.

Confusion Matrix:
    - Computes the confusion matrix based on the ground truth and predicted labels.

Confusion Matrix Plotting:
    - Plots the confusion matrix and saves the plot as 'confusion_matrix.png'.

Example Usage:
    - Ensure the 'model.h5' file and required data paths are correctly set.
    - Run the script to evaluate the model and visualize the confusion matrix.
"""

from src.data import test_ds, CSV_DICT, TEST_SET_PATH
import tensorflow as tf
from src.custom_objects.layer_scale import LayerScale
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(CSV_DICT[TEST_SET_PATH])
class_col = df['Bald']
class_col = class_col.tolist()
model = tf.keras.models.load_model(
    '../../tensorboard/best_model_continued_2/model.keras',
    custom_objects={'LayerScale': LayerScale})
y_pred = (model.predict(test_ds) > 0.5).astype('int')

cm = confusion_matrix(class_col, y_pred.reshape((-1,)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('confusion_matrix_best_continued_2')
plt.show()
