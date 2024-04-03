import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers.legacy import Adam
from keras import metrics

(X, Y), (X_, Y_) = keras.datasets.mnist.load_data()

x_train_all , x_test , y_train_all , y_test = train_test_split(X, Y , test_size=1/12, random_state=0) # 5000 validation

value_to_train_for = 5
sample_size = 55000
# sample_size = sample_size_list[i]
batch_size = int(sample_size/100)
epc = 50

while True:
    rnd_idx = np.random.choice(x_train_all.shape[0], sample_size, replace=False)
    x_train = np.take(x_train_all, rnd_idx, axis=0)
    y_train = np.take(y_train_all, rnd_idx, axis=0)
    if value_to_train_for in y_train:
        break

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
X_ = X_.reshape(-1,28,28,1)

x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
X_ = X_.astype("float32")/255

y_train = y_train == value_to_train_for
y_test = y_test == value_to_train_for
Y_ = Y_ == value_to_train_for

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[metrics.BinaryAccuracy(),metrics.AUC(), metrics.Precision(), metrics.Recall(), metrics.TruePositives(), metrics.FalsePositives()])
model.fit(x_train, y_train, batch_size = batch_size, epochs = epc, validation_data = (x_test, y_test))

ture_loss, true_acc, true_auc, true_precision, true_recall, true_TP, true_FP = model.evaluate(X_,Y_, batch_size=Y_.shape[0])

print("accuracy:", true_acc)
print("precision:", true_precision)

