import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers.legacy import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras import datasets
from keras import metrics
from keras import backend
from sklearn.model_selection import train_test_split

# epoch_list = [1,3]
epoch_list = [1,3,5,10,30]
# batch_list = [1,10,100]
# batch_list = [1]
nbr_runs = 3
sample_size_list = [100, 1000, 5000]
loss_mean_list_epc=[]
loss_std_list_epc=[]
acc_mean_list_epc=[]
acc_std_list_epc=[]
auc_mean_list_epc=[]
auc_std_list_epc=[]
precision_mean_list_epc=[]
precision_std_list_epc=[]
recall_mean_list_epc=[]
recall_std_list_epc=[]
tp_mean_list_epc=[]
tp_std_list_epc=[]
fp_mean_list_epc=[]
fp_std_list_epc=[]

for epc in epoch_list:
# for btch in batch_list:

    loss_mean_list=[]
    loss_std_list=[]
    acc_mean_list=[]
    acc_std_list=[]
    auc_mean_list=[]
    auc_std_list=[]
    precision_mean_list=[]
    precision_std_list=[]
    recall_mean_list=[]
    recall_std_list=[]
    tp_mean_list=[]
    tp_std_list=[]
    fp_mean_list=[]
    fp_std_list=[]

    for i in range (len(sample_size_list)):

        loss_list = []
        acc_list = []
        auc_list = []
        precision_list = []
        recall_list = []
        tp_list = []
        fp_list = []

        for j in range(nbr_runs):

            backend.clear_session()
            # seed = 0
            # np.random.seed(seed)

            (X, Y), (X_, Y_) = datasets.mnist.load_data()

            #splitting dataframe using train_test_split
            x_train_all , x_test , y_train_all , y_test = train_test_split(X, Y , test_size=1/12, random_state=0) # 5000 validation
            # x_train_all , x_test , y_train_all , y_test = train_test_split(X, Y , test_size=1/12) # 5000 validation

            value_to_train_for = 5
            sample_size = sample_size_list[i]
            # batch_size = sample_size//btch
            batch_size = sample_size//50
            # batch_size = 100
            epochs = epc

            # count = 0
            while True:
                rnd_idx = np.random.choice(x_train_all.shape[0], sample_size, replace=False)
                x_train = np.take(x_train_all, rnd_idx, axis=0)
                y_train = np.take(y_train_all, rnd_idx, axis=0)
                if value_to_train_for in y_train:
                    break

            #first param in reshape is number of examples. We can pass -1 here as we want numpy to figure that out by itself
            #reshape(examples, height, width, channels)
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)
            X_ = X_.reshape(-1,28,28,1)

            datagen = ImageDataGenerator(
                        featurewise_center=False,  # set input mean to 0 over the dataset
                        samplewise_center=False,  # set each sample mean to 0
                        featurewise_std_normalization=False,  # divide inputs by std of the dataset
                        samplewise_std_normalization=False,  # divide each input by its std
                        zca_whitening=False,  # apply ZCA whitening
                        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                        zoom_range = 0.1, # Randomly zoom image
                        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                        horizontal_flip=False,  # randomly flip images
                        vertical_flip=False)  # randomly flip images

            #convert values to float as result will be a float. If not done vals are set to zero
            x_train = x_train.astype("float32")/255
            x_test = x_test.astype("float32")/255
            X_ = X_.astype("float32")/255

            #fitting the ImageDataGenerator we defined above
            datagen.fit(x_train)

            y_train = y_train == value_to_train_for
            y_test = y_test == value_to_train_for
            Y_ = Y_ == value_to_train_for

            #Conv2d data_format parameter we use 'channel_last' for imgs
            model = Sequential()

            model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last',
                            input_shape=(28,28,1)))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid' ))
            model.add(Dropout(0.25))

            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))
            model.add(Dense(1024, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            #model.add(Dense(10, activation='softmax'))
            model.add(Dense(1, activation='sigmoid'))

            #Optimizer
            optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

            #Compiling the model
            model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[metrics.BinaryAccuracy(),metrics.AUC(), metrics.Precision(), metrics.Recall(), metrics.TruePositives(), metrics.FalsePositives()])

            # model.summary()

            #for our case LearningRateScheduler will work great
            reduce_lr = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

            #by default this is evaluated on 'val_loss'
            early_stopping = EarlyStopping(
                min_delta=0.001, # minimium amount of change to count as an improvement
                patience=20, # how many epochs to wait before stopping
                restore_best_weights=True,
            )

            # Fit the Model
            # history = model.fit(datagen.flow(x_train, y_train, batch_size = batch_size), epochs = epochs, validation_data = (x_test, y_test), verbose=1)

            # Fit the Model
            history = model.fit(datagen.flow(x_train, y_train, batch_size = batch_size), epochs = epochs,
                                        validation_data = (x_test, y_test), verbose=1,
                                        # steps_per_epoch=x_train.shape[0] // batch_size,
                                        callbacks = [reduce_lr]) #left out early_stopping parameter as it gets better accuracy

            ture_loss, true_acc, true_auc, true_precision, true_recall, true_TP, true_FP = model.evaluate(X_,Y_, batch_size=Y_.shape[0])

            loss_list.append(ture_loss)
            acc_list.append(true_acc)
            auc_list.append(true_auc)
            precision_list.append(true_precision)
            recall_list.append(true_recall)
            tp_list.append(true_TP)
            fp_list.append(true_FP)
            
            # if os.path.exists("/results.txt"):
            f = open("results.txt","a")
            f.write("\n")
            f.write("%d\t" % (j))
            f.write("%d\t" % (sample_size))
            f.write("%d\t" % (batch_size))
            f.write("%d\t" % (epochs))
            f.write("%.4f\t" % (ture_loss))
            f.write("%.4f\t" % (true_acc))
            f.write("%.2f\t" % (true_precision))
            f.write("%.2f\t" % (true_recall))
            f.write("%.2f\t" % (true_auc))
            f.write("%.2f\t" % (true_TP))
            f.write("%.2f\t" % (true_FP))
            f.close()

        loss_mean_list.append(np.mean(loss_list))
        loss_std_list.append(np.std(loss_list))
        acc_mean_list.append(np.mean(acc_list))
        acc_std_list.append(np.std(acc_list))
        auc_mean_list.append(np.mean(auc_list))
        auc_std_list.append(np.std(auc_list))
        precision_mean_list.append(np.mean(precision_list))
        precision_std_list.append(np.std(precision_list))
        recall_mean_list.append(np.mean(recall_list))
        recall_std_list.append(np.std(recall_list))
        tp_mean_list.append(np.mean(tp_list))
        tp_std_list.append(np.std(tp_list))
        fp_mean_list.append(np.mean(fp_list))
        fp_std_list.append(np.std(fp_list))

    loss_mean_list_epc.append(loss_mean_list)
    loss_std_list_epc.append(loss_std_list)
    acc_mean_list_epc.append(acc_mean_list)
    acc_std_list_epc.append(acc_std_list)
    auc_mean_list_epc.append(auc_mean_list)
    auc_std_list_epc.append(auc_std_list)
    precision_mean_list_epc.append(precision_mean_list)
    precision_std_list_epc.append(precision_std_list)
    recall_mean_list_epc.append(recall_mean_list)
    recall_std_list_epc.append(recall_std_list)
    tp_mean_list_epc.append(tp_mean_list)
    tp_std_list_epc.append(tp_std_list)
    fp_mean_list_epc.append(fp_mean_list)
    fp_std_list_epc.append(fp_std_list)

plot_insert_list = sample_size_list*len(epoch_list)
plot_mean_list = np.concatenate([i1 for i1 in acc_mean_list_epc]).tolist()
plot_detail_list = np.concatenate([['Epoch='+ str(i) for j in range(len(sample_size_list))] for i in epoch_list]).tolist()
plot_std_list = np.concatenate([i2 for i2 in acc_std_list_epc]).tolist()

df1 = pd.DataFrame({
    'insert': plot_insert_list,
    'mean': plot_mean_list,
    'detail': plot_detail_list, 
    'std': plot_std_list})

fig1, ax1 = plt.subplots()

for key, group in df1.groupby('detail'):
    group.plot('insert', 'mean', yerr='std', 
               label=key, ax=ax1,ylim=(0, 1))

plt.savefig('fig_acc.png')

plot_insert_list = []
plot_mean_list = []
plot_detail_list = []
plot_std_list = []

plot_insert_list = sample_size_list*len(epoch_list)
plot_mean_list = np.concatenate([i1 for i1 in precision_mean_list_epc]).tolist()
plot_detail_list = np.concatenate([['Epoch='+ str(i) for j in range(len(sample_size_list))] for i in epoch_list]).tolist()
plot_std_list = np.concatenate([i2 for i2 in precision_std_list_epc]).tolist()

df2 = pd.DataFrame({
    'insert': plot_insert_list,
    'mean': plot_mean_list,
    'detail': plot_detail_list, 
    'std': plot_std_list})

fig2, ax2 = plt.subplots()

for key, group in df2.groupby('detail'):
    group.plot('insert', 'mean', yerr='std', 
               label=key, ax=ax2,ylim=(0, 1))

plt.savefig('fig_precision.png')


# plot_insert_list = sample_size_list*3
# plot_mean_list = acc_mean_list+precision_mean_list+recall_mean_list
# plot_detail_list = ['Accuracy']*len(sample_size_list) + ['Precision']*len(sample_size_list) + ['Recall']*len(sample_size_list)
# plot_std_list = acc_std_list+precision_std_list+recall_std_list

# df1 = pd.DataFrame({
#     'insert': plot_insert_list,
#     'mean': plot_mean_list,
#     'detail': plot_detail_list, 
#     'std': plot_std_list})

# fig, ax = plt.subplots()
 
# for key, group in df1.groupby('detail'):
#     group.plot('insert', 'mean', yerr='std', 
#                label=key, ax=ax,ylim=(0, 1))

# plt.savefig('foo1.png')

# plot_insert_list = sample_size_list*2
# plot_mean_list = tp_mean_list+fp_mean_list
# plot_detail_list = ['TP']*len(sample_size_list)+ ['FP']*len(sample_size_list)
# plot_std_list = tp_std_list+fp_std_list

# df2 = pd.DataFrame({
#     'insert': plot_insert_list,
#     'mean': plot_mean_list,
#     'detail': plot_detail_list, 
#     'std': plot_std_list})

# fig, ax = plt.subplots()
 
# for key, group in df2.groupby('detail'):
#     group.plot('insert', 'mean', yerr='std', 
#                label=key, ax=ax)

# plt.savefig('foo2.png')


