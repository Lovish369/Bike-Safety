import time
st_lib = time.time()
import pandas as pd
from scipy import stats
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import os, psutil
et_lib = time.time()
elp_lib = et_lib - st_lib
print('Importing lib in:', elp_lib, ' sec')


st_code = time.time()
test =pd.read_csv('/home/pi/Desktop/lstm/data_test_24-29.csv')
train = pd.read_csv('/home/pi/Desktop/lstm/data_train_24-29.csv')

#print(train.head)

train_data = train.drop(['TIME IN GMT','TIME IN IST','Lat ','Long', 'Time','T','Date'], axis=1)
test_data = test.drop(['TIME IN GMT','TIME IN IST','Lat ','Long', 'Time','T','Date'], axis=1)

#print(train_data.head)

train_data.dropna(axis=0, how='any', inplace=True)
#print(train_data.shape)

test_data.dropna(axis=0, how='any', inplace=True)
#print(test_data.shape)

n_time_steps = 104
n_features = 7 
step = 104
n_classes = 5 
n_epochs = 50       
batch_size = 64   
learning_rate = 0.0001
l2_loss = 0.0015

segments = []
labels = []

for i in range(0,  train_data.shape[0]- n_time_steps, step):  

    Ax_tr = train_data['Ax'].values[i: i + n_time_steps]

    Ay_tr = train_data['Ay'].values[i: i + n_time_steps]

    Az_tr = train_data['Az'].values[i: i + n_time_steps]

    Gx_tr = train_data['Gx'].values[i: i + n_time_steps]

    Gy_tr = train_data['Gy'].values[i: i + n_time_steps]

    Gz_tr = train_data['Gz'].values[i: i + n_time_steps]

    Speed_tr = train_data['Speed'].values[i: i + n_time_steps]


    label_tr = stats.mode(train_data['Label'][i: i + n_time_steps])[0][0]

    segments.append([Ax_tr, Ay_tr, Az_tr, Gx_tr,Gy_tr,Gz_tr,Speed_tr])

    labels.append(label_tr)

X_train = np.asarray(segments, dtype= np.float32).reshape(-1, n_time_steps, n_features)

y_train = np.asarray(pd.get_dummies(labels), dtype = np.float32)

#print(X_train.shape)

reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, n_time_steps, n_features)

labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

#print(reshaped_segments.shape)

X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size = 0.25)

#print(X_test.shape)
#print(y_test.shape)

model1 = load_model('/home/pi/Desktop/lstm/model_bike')

st_predict = time.time()
predictions = model1.predict(X_test)
stop_predict = time.time()
tt_predict = stop_predict - st_predict
print('Prediction Time: ', tt_predict, 'sec')

loss, accuracy = model1.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
print("Test Accuracy :", accuracy)
print("Test Loss :", loss)
st_code_end = time.time()
elp_code = st_code_end - st_code
print('Code takes:', elp_code, ' sec to run')

process = psutil.Process(os.getpid())
print('Program takes',process.memory_info().rss / 1024 ** 2, 'mega-bytes of RAM')

class_labels = ['BUMP', 'LEFT',  'RIGHT','STOP', 'STRAIGHT']
max_test = np.argmax(y_test, axis=1)
max_predictions = np.argmax(predictions, axis=1)
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)

sns.heatmap(confusion_matrix, xticklabels = class_labels, yticklabels = class_labels, annot = True, linewidths = 0.1, fmt='d', cmap = 'YlGnBu')
plt.title("Confusion matrix", fontsize = 15)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


class_labels = ['BUMP', 'LEFT',  'RIGHT','STOP', 'STRAIGHT']
max_test = np.argmax(y_test, axis=1)
max_predictions = np.argmax(predictions, axis=1)
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)
cmn = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(cmn, xticklabels = class_labels, yticklabels = class_labels, annot = True, linewidths = 0.1, fmt='2f', cmap = 'YlGnBu')
plt.title("Confusion matrix", fontsize = 15)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()