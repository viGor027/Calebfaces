from data import test_ds, CSV_DICT, TEST_SET_PATH
import tensorflow as tf
from layer_scale import LayerScale
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(CSV_DICT[TEST_SET_PATH])
class_col = df['Bald']
class_col = class_col.tolist()
model = tf.keras.models.load_model('best_model/model.h5', custom_objects={'LayerScale': LayerScale})
y_pred = (model.predict(test_ds) > 0.5).astype('int')

cm = confusion_matrix(class_col, y_pred.reshape((-1,)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('confusion_matrix')
plt.show()
