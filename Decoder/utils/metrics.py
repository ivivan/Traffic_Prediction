import numpy as np
import math
import scipy
from sklearn.metrics import mean_squared_log_error



def rrse_(y_true, y_pred):
  return np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def CORR(y_true, y_pred):
  N = y_true.shape[0]
  total = 0.0
  for i in range(N):
    if math.isnan(scipy.stats.pearsonr(y_true[i], y_pred[i])[0]):
      N -= 1
    else:
      total += scipy.stats.pearsonr(y_true[i], y_pred[i])[0]
  return total / N

def mean_absolute_percentage_error(y_true, y_pred):
  """
  Use of this metric is not recommended; for illustration only.
  See other regression metrics on sklearn docs:
    http://scikit-learn.org/stable/modules/classes.html#regression-metrics
  Use like any other metric
  >>> y_true = [3, -0.5, 2, 7]; y_pred = [2.5, -0.3, 2, 8]
  >>> mean_absolute_percentage_error(y_true, y_pred)
  Out[]: 24.791666666666668
  """

  # y_true, y_pred = check_arrays(y_true, y_pred)

  ## Note: does not handle mix 1d representation
  # if _is_1d(y_true):
  #    y_true, y_pred = _check_1d_array(y_true, y_pred)

  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def root_relative_squared_error(y_true, y_pred):
  mn = np.mean(y_true)
  return np.sqrt(np.average((y_true - y_pred) ** 2)) / np.sqrt(np.average((y_true - mn) ** 2))

# Root Mean Squared Logarithmic Error (RMSLE)
# def rmsle(y, y0):
#   return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))
def RMSLE(y_true, y_pred):
  return np.sqrt(mean_squared_log_error(y_true,y_pred))