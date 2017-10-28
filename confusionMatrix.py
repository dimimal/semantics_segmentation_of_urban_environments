# Implementation about confusion matrices to apply onto 
# keras models

from sklearn.metrics import confusion_matrix
import numpy as np

y_true = np.array([0] * 1000 + [1] * 1000)
y_pred = y_true > 0.5

print confusion_matrix(y_true, y_pred)
