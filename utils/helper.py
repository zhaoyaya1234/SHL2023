import pandas as pd 
import numpy as np
import utm
from sklearn.metrics import precision_score, confusion_matrix, classification_report,roc_auc_score
import seaborn as sn
import matplotlib.pyplot as plt

label_dict = {1: 'Still', 2: 'Walking', 3: 'Run', 4: 'Bike', 5: 'Car', 6: 'Bus', 7: 'Train', 8: 'Subway'}

# (latitude, longitude) -> east
def gps2utm_east(x):
    try:
        return utm.from_latlon(x['latitude'], x['longitude'])[0]
    except:
        return np.nan
# (latitude, longitude) -> north
def gps2utm_north(x):
    try:
        return utm.from_latlon(x['latitude'], x['longitude'])[1]
    except:
        return np.nan



# evaluate prediction result
def evaluate(y_true, y_pred, normalize = True, names = list(label_dict.values())):
    if normalize:
        conf = confusion_matrix(y_true, y_pred, normalize = 'true')
    else:
        conf = confusion_matrix(y_true, y_pred)
    print(confusion_matrix(y_true, y_pred))
    sn.heatmap(conf)
    print(classification_report(y_true, y_pred, target_names = names))

# save prediction result
def save_prediction(pred_time, pred_res, file_path = 'data2023/test/RY_predictions.txt'):
    res = pd.DataFrame({'time': pred_time, 'label': pred_res})
    res.to_csv(file_path, index = False, header = False, sep = '\t')

# plot prediction labels
def plot_prediction(y_pred, y_true):
    plt.figure(figsize = [20, 8])
    plt.plot(y_pred, alpha = 0.4)
    plt.plot(y_true)


def plot_box(y, pred):
    plt.figure(figsize = [12, 10])
    sn.heatmap(confusion_matrix(y, pred), cmap = 'Blues', annot = True, fmt = 'd', xticklabels = list(label_dic.values()),
    yticklabels = list(label_dict.values()))