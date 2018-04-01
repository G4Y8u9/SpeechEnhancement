# -*- coding: utf-8 -*-

from keras import backend as K
from keras import optimizers, regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from scipy.io import wavfile
from scipy.signal import stft, istft
from util import get_Zyy, list_dir_shuffle, make_window_buffer, test_model, unnormalize
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time


"""
    parameters used in train
    now we only have AWGN in -5, 5, 10dB, will add new noise type later...
    :parameter testdB: -5, 5, 10 available
    :parameter fr: the number of frames per frame group for DNN input
"""
testdB = 5
fr = 9
normal_flag = 1


"""
    sth useful
    !If you change these four parameters, you must RETRAIN the model!
    :parameter freq: sample rate
    :parameter nffts: fft size
    :parameter hid_neus: number of neurons per hidden layer
    :parameter neighbor: used in frame group construction, see 'make_window_buffer' method in util.py
    :parameter frwd: size of stft shape[0]
"""
freq = 16000
nffts = 256
hid_neus = 2048
neighbor = int((fr-1)/2)
frwd = int(nffts/2+1)


"""
    make sure the changes of the parameters is correct
"""
assert (testdB == 10 or testdB == 5 or testdB == -5), 'testdB must be -5, 5 or 10'
assert (nffts == 128 or nffts == 256 or nffts == 512 or nffts == 1024 or nffts == 2048 or nffts == 4096),\
    'nffts must be 2^n, but now it is ' + str(nffts)
assert (fr == 1 or fr == 5 or fr == 9 or fr == 11 or fr == 13), 'fr must be 1, 5, 9, 11, 13'
assert (hid_neus >= frwd * fr), 'Hidden neurons are too few! Reduce fr or reduce nffts or increase hid_neus!'


"""
    load train and test data
"""
X_train_list = []
Y_train_list = []
X_test_list = []
Y_test_list = []
# these two func make sure the data loaded is picked randomly
list_dir_shuffle('E:\\trainData\\newdata\\' + str(testdB) + 'db\\noisy',
                 'E:\\trainData\\newdata\\' + str(testdB) + 'db\\pure',
                 X_train_list, Y_train_list, 1000)
list_dir_shuffle('E:\\trainData\\newdata\\' + str(testdB) + 'db\\noisy',
                 'E:\\trainData\\newdata\\' + str(testdB) + 'db\\pure',
                 X_test_list, Y_test_list, 2)


"""
    construct model
    model is self-adaptive to the parameters, so no need to change
"""
model = Sequential()
# input layer to hidden layer 1   frame * framewidth float -> hidden neurons float
model.add(Dense(hid_neus, activation='relu', dtype=tf.float32, input_dim=frwd*fr))
# model.add(Dropout(0.1))
# hidden layer 2   hidden neurons float -> hidden neurons float
model.add(Dense(hid_neus, activation='relu', dtype=tf.float32))
# hidden layer 3   hidden neurons float -> hidden neurons float
model.add(Dense(hid_neus, activation='relu', dtype=tf.float32))
# output layer   hidden neurons float -> framewidth float
model.add(Dense(frwd, activation='linear', dtype=tf.float32))


"""
    Three Optimizer func used in test, but only sgd is used
"""
sgd = optimizers.SGD(lr=0.0001)
adam = optimizers.Adam(lr=0.0001)
mo = optimizers.TFOptimizer(tf.train.MomentumOptimizer(0.005, 0.9))


def mean_squared_error_divN(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)/(frwd*fr)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


model.compile(
    optimizer=adam,
    loss=mean_squared_error,
    # metrics=['mse']
)


"""
    training
"""
epochs = 1   # total epoch
start_time = time.time()
for epoch in range(epochs):
    nfile = 0
    for x, y in zip(X_train_list, Y_train_list):
        xt = make_window_buffer(x, neighbor=neighbor, nfft=nffts, normal_flag=normal_flag)
        yt = get_Zyy(y, nfft=nffts, normal_flag=normal_flag)
        model.fit(xt, yt)
        print('EPOCH %d/%d, file %d' % (epoch+1, epochs, nfile+1))
        nfile += 1
        if nfile % 50 == 0:
            model.save_weights('.\\models_allkind\\myModelWeight_exp_fr' + str(fr)
                               + ('' if normal_flag == 0 else '_norm_') + '_SNR' + str(testdB) + '.h5')
            print('.\\models_allkind\\myModelWeight_exp_fr' + str(fr)
                  + ('' if normal_flag == 0 else '_norm_') + '_SNR' + str(testdB) + '.h5')
    del nfile
print(time.time()-start_time)


"""
    save model
"""
model.save_weights('.\\models_allkind\\myModelWeight_exp_fr' + str(fr)
                   + ('' if normal_flag == 0 else '_norm_') + '_SNR' + str(testdB) + '.h5')
print('model saved in ' + '.\\models_allkind\\myModelWeight_exp_fr' + str(fr)
      + ('' if normal_flag == 0 else '_norm_') + '_SNR' + str(testdB) + '.h5')

# """
#     load model
# """
# # model.load_weights('.\\models_allkind\\myModelWeight_exp_fr'+str(fr)+'_SNR_'+str(testdB)+'.h5')
# model.load_weights('.\\models_allkind\\myModelWeight_exp_norm_fr9_SNR_5.h5')


# """
#     Test model
# """
# print('\n-----------------TESTING-----------------')
# for x, y in zip(X_test_list, Y_test_list):
#     print('testing file from: ', x)
#     xt = make_window_buffer(x, neighbor=neighbor, nfft=nffts)
#     yt = get_Zyy(y, nfft=nffts)
#     loss = model.evaluate(xt, yt)
#     print('loss: ', loss)


"""
    now we use the model to do sth
"""
testDir = 'E:\\SpeechEnhancement\\test\\5db\\06.wav'
_, s = wavfile.read(testDir)
f, t, Zxx = stft(s, freq)
Zxx1 = np.log((np.abs(Zxx)).T+1e-7)
# construct the input format for the DNN
y_input = make_window_buffer(testDir, neighbor=neighbor, nfft=nffts)
y = model.predict(y_input).T
# construct the spectrogram with the abs of the output and the angle of the noisy sound, then ISTFT
# ypreComplex = y * np.exp(complex(0, 1) * np.angle(Zxx))
ypreComplex = np.exp(y) * np.exp(complex(0, 1) * np.angle(Zxx)) if normal_flag == 0 \
        else np.exp(unnormalize(y, Zxx1)) * np.exp(complex(0, 1) * np.angle(Zxx))
_, xrec = istft(ypreComplex, freq)
# the DNN output spectrogram
plt.figure()
plt.pcolormesh(t, f, np.abs(ypreComplex))
plt.ylim([f[1], f[-1]])
plt.title('DNN output')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
# the noisy speech spectrogram
plt.figure()
plt.pcolormesh(t, f, np.abs(Zxx1))
plt.ylim([f[1], f[-1]])
plt.title('noisy spec')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
# # and take a look of pure speech spectrogram
# _, pure = wavfile.read('E:\\SpeechEnhancement\\test\\5db\\06.wav')
# _, _, pureX = stft(pure, freq)
# plt.figure()
# plt.pcolormesh(t, f, np.abs(pureX))
# plt.ylim([f[1], f[-1]])
# plt.title('pure spec')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
# write wav file
dataWrite = xrec.astype(np.int16)   # extremely important!
wavfile.write('./predict777' + str(testdB) + '.wav', freq, dataWrite)


"""
    just some other tests
"""
test_model(model, 'E:\\SpeechEnhancement\\test\\5db\\01.wav', 'E:\\SpeechEnhancement\\test\\5db\\output_norm0_01.wav',
           neighbor=neighbor, nffts=nffts, normal_flag=normal_flag)
test_model(model, 'E:\\SpeechEnhancement\\test\\5db\\02.wav', 'E:\\SpeechEnhancement\\test\\5db\\output_norm0_02.wav',
           neighbor=neighbor, nffts=nffts, normal_flag=normal_flag)
test_model(model, 'E:\\SpeechEnhancement\\test\\5db\\03.wav', 'E:\\SpeechEnhancement\\test\\5db\\output_norm0_03.wav',
           neighbor=neighbor, nffts=nffts, normal_flag=normal_flag)
test_model(model, 'E:\\SpeechEnhancement\\test\\5db\\04.wav', 'E:\\SpeechEnhancement\\test\\5db\\output_norm0_04.wav',
           neighbor=neighbor, nffts=nffts, normal_flag=normal_flag)
test_model(model, 'E:\\SpeechEnhancement\\test\\5db\\05.wav', 'E:\\SpeechEnhancement\\test\\5db\\output_norm0_05.wav',
           neighbor=neighbor, nffts=nffts, normal_flag=normal_flag)
