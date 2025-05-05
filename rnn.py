import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import json

class NeuralNetMLP(object):
    def __init__(self, epochs=100, minibatch_size=100, seed=1):

        self.random = np.random.RandomState(seed)
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=50, input_shape=(100, 1), return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))  # Added Dropout layer here
        model.add(tf.keras.layers.LSTM(units=50))
        model.add(tf.keras.layers.Dense(units=1, activation='relu'))

        model.summary()

        ## compile:
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=[self.custom_accuracy])
        self.model = model


    def predict(self, X):
        return self.model.predict(X)
    
    def custom_accuracy(self, y_true, y_pred):
        return tf.reduce_mean(tf.cast(tf.abs(y_true[:,0] - y_pred[:,0]) < 0.02, tf.float32))
    
    # Custom callback to print progress
    class PrintProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, father):
            super(NeuralNetMLP.PrintProgressCallback, self).__init__()
            self.father = father
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            train_loss = logs.get('loss')
            val_loss = logs.get('val_loss')
            train_acc = logs.get('custom_accuracy') * 100
            val_acc = logs.get('val_custom_accuracy') * 100
            epoch_strlen = len(str(self.father.epochs))
            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, epoch+1, self.father.epochs, train_loss,
                              train_acc, val_acc))
            sys.stderr.flush()
            if train_acc > val_acc:
                self.father.over_fit_cnt += 1
            else:
                self.father.over_fit_cnt = 0
            if (epoch+1) % 100 == 0:
                self.father.shold_save = 1

            #if (self.father.shold_save > 0 and val_acc > train_acc) or val_acc > 90:  # To privent from over fitting.
            if val_acc > train_acc or self.father.shold_save > 0:
                model_file = "rnn_50_50_LSTM" + "{0:4d}_{1:.2f}-{2:.2f}={3:.2f}.h5".format(epoch+1, val_acc, train_acc, val_acc-train_acc)
                model_dir = os.path.join('.', 'models', 'rnn-50-50-LSTM-paper2')
                model_file_path = os.path.join(model_dir, model_file)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                if val_acc > self.father.max_acc:                
                    self.father.model.save(model_file_path)
                    self.father.max_acc = val_acc
                    self.father.shold_save = 2
                elif val_acc-train_acc > self.father.max_space:
                    self.father.model.save(model_file_path)
                    self.father.max_space = val_acc-train_acc
                    self.father.shold_save = 2
                elif self.father.shold_save > 0:
                    self.father.model.save(model_file_path)
                    self.father.shold_save -= 1
            
    
                # Check if over_fit_cnt exceeds 3 and stop training
            if self.father.over_fit_cnt > 5:
                self.father.over_fit_cnt = 0
                #self.model.stop_training = True
                print(f"\nOverfitting detected. Training stopped and model saved. {epoch}")

    def fit(self, X_train, y_train, X_valid, y_valid):
        # Reshape input data for RNN
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))

        self.over_fit_cnt=0
        self.shold_save=0
        self.max_acc=0
        self.max_space=0

        hist = self.model.fit(X_train, y_train, 
                              validation_data=(X_valid, y_valid), 
                              epochs=self.epochs, batch_size=self.minibatch_size, verbose=0,
                              callbacks=[self.PrintProgressCallback(self)])

        self.history = hist.history
        #self.model.save('rnn_50_50GRU_100steps.h5')

        return self

