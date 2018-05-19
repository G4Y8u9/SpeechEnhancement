import tensorflow as tf
import numpy as np
import os
import cv2
from keras import backend as K
from scipy.io import wavfile
from scipy.signal import stft, istft
from numpy import log, exp, abs as abss, angle as ang, zeros as zer
from python_speech_features import mfcc, delta
freq = 16000


def list_dir_shuffle(path1, path2, list1, list2, num=-1):
    """
    get the list of all paths of every file in 'path1' and 'path2'
    then shuffle them together to make sure the X and Y are still binding
    :param path1: dir path1
    :param path2: dir path2
    :param list1:
    :param list2:
    :param num: how many file do you want
    """
    temp1, temp2 = os.listdir(path1), os.listdir(path2)
    import sklearn.utils as sku
    temp1, temp2 = sku.shuffle(temp1, temp2)
    for file1, file2 in zip(temp1[0:num], temp2[0:num]):
        file_path1 = os.path.join(path1, file1)
        file_path2 = os.path.join(path2, file2)
        list1.append(file_path1)
        list2.append(file_path2)


def make_window_buffer(xdir, neighbor=2, nfft=256, normal_flag=0):
    """
    get frame group for DNN input, and this is the key
    expend every row of the array to the combination of itself and its neighbors

    Example:
    given array like this:
                 [[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3],
                  [4, 4, 4],
                  [5, 5, 5],
                  [6, 6, 6],
                  [7, 7, 7]]
    and we combine 1 neighbor, then we have:
        [[1, 1, 1, 1, 1, 1, 2, 2, 2],
         [1, 1, 1, 2, 2, 2, 3, 3, 3],
         [2, 2, 2, 3, 3, 3, 4, 4, 4],
         [3, 3, 3, 4, 4, 4, 5, 5, 5],
         [4, 4, 4, 5, 5, 5, 6, 6, 6],
         [5, 5, 5, 6, 6, 6, 7, 7, 7],
         [6, 6, 6, 7, 7, 7, 7, 7, 7]]
        (neighbor)↑,↑,↑ (neighbor)
    noticing this column marked above, they are the origin array
    noticing that for the start and end, it will repeat 'neighbour' times to make up
    :param xdir:
    :param neighbor:
    :param nfft:
    :param normal_flag: 0 for log power, 1 for normalized log power
    :return:
    """
    _, x = wavfile.read(xdir)
    _, _, Zxx = stft(x, freq, nfft=nfft)
    Zxx = log((abss(Zxx)).T+1e-7) if normal_flag == 0 else normalize_mean(log((abss(Zxx)).T+1e-7))
    m, n = Zxx.shape
    tmp = zer(m * n * (neighbor * 2 + 1), dtype='float32').reshape(m, -1)
    for i in range(2 * neighbor + 1):
        if i <= neighbor:
            shift = neighbor - i
            tmp[shift:m, i * n: (i + 1) * n] = Zxx[:m - shift]
            for j in range(shift):
                tmp[j, i * n: (i + 1) * n] = Zxx[0, :]
        else:
            shift = i - neighbor
            tmp[:m-shift, i * n: (i+1) * n] = Zxx[shift:m]
            for j in range(shift):
                tmp[m-(j + 1), i * n: (i + 1) * n] = Zxx[m-1, :]
    return tmp


def combine_with_mfcc(xdir, neighbor=2, nfft=256, normal_flag=0):
    _, x = wavfile.read(xdir)
    _, _, Zxx = stft(x, freq, nfft=nfft)
    Zxx = log((abss(Zxx)).T + 1e-7) if normal_flag == 0 else normalize_mean(log((abss(Zxx)).T + 1e-7))
    m, n = Zxx.shape
    tmp = zer(m * n * (neighbor * 2 + 1), dtype='float32').reshape(m, -1)
    for i in range(2 * neighbor + 1):
        if i <= neighbor:
            shift = neighbor - i
            tmp[shift:m, i * n: (i + 1) * n] = Zxx[:m - shift]
            for j in range(shift):
                tmp[j, i * n: (i + 1) * n] = Zxx[0, :]
        else:
            shift = i - neighbor
            tmp[:m - shift, i * n: (i + 1) * n] = Zxx[shift:m]
            for j in range(shift):
                tmp[m - (j + 1), i * n: (i + 1) * n] = Zxx[m - 1, :]
    # now tmp is "make_window_buffer" output
    # then calc mfcc & d & dd
    mfcc_data = combine_mfcc_d_dd(mfcc(x, freq, winlen=0.016, winstep=0.008, nfft=256, winfunc=np.bartlett))
    while True:
        try:
            tmp1 = np.concatenate((tmp, mfcc_data), axis=1)
            break
        except ValueError:
            mfcc_data = np.concatenate((mfcc_data, np.zeros([1, mfcc_data.shape[1]])), axis=0)
            continue
    return tmp1


def combine_mfcc_d_dd(mfcc_data):
    assert (mfcc_data.shape[1] == 13), 'mfcc_data.shape[1] must be 13'
    d = delta(mfcc_data, 2)
    dd = delta(d, 2)
    return np.concatenate((mfcc_data, d, dd), axis=1)


def get_Zyy(ydir, nfft=256, normal_flag=0):
    """
    get the stft.T of the wav file in ydir
    usually used in train and test, not in practical application
    :param ydir:
    :param nfft:
    :param normal_flag: 0 for log power, 1 for normalized log power
    :return:
    """
    _, y = wavfile.read(ydir)
    _, _, Zyy = stft(y, freq, nfft=nfft)
    return log((abss(Zyy)).T+1e-7) if normal_flag == 0 else normalize_mean(log((abss(Zyy)).T+1e-7))


def combine_with_mfcc_Zyy(ydir, nfft=256, normal_flag=0):
    _, y = wavfile.read(ydir)
    _, _, Zyy = stft(y, freq, nfft=nfft)
    y_data = log((abss(Zyy)).T + 1e-7) if normal_flag == 0 else normalize_mean(log((abss(Zyy)).T + 1e-7))
    mfcc_data = combine_mfcc_d_dd(mfcc(y, freq, winlen=0.016, winstep=0.008, nfft=256, winfunc=np.bartlett))
    while True:
        try:
            tmp1 = np.concatenate((y_data, mfcc_data), axis=1)
            break
        except ValueError:
            mfcc_data = np.concatenate((mfcc_data, np.zeros([1, mfcc_data.shape[1]])), axis=0)
            continue
    return tmp1


def normalize_mean(x):
    # normalize x
    return (x-x.mean())/x.std()


def unnormalize(x1, x2):
    # unnormalize x1 by the std and mean of x2
    return x1*x2.std()+x2.mean()


def test_model(model, input_path, output_path, neighbor, nffts, normal_flag=0):
    _, s = wavfile.read(input_path)
    _, _, Zxx = stft(s, freq)
    Zxx1 = log((abss(Zxx)).T+1e-7)
    y_input = make_window_buffer(input_path, neighbor=neighbor, nfft=nffts, normal_flag=normal_flag)
    y = model.predict(y_input).T
    # print(y.shape, unnormalize(y, abss(Zxx)).shape, unnormalize(y, abss(Zxx)).dtype)
    ypreComplex = exp(y) * exp(complex(0, 1) * ang(Zxx)) if normal_flag == 0 \
        else exp(unnormalize(y, Zxx1)) * exp(complex(0, 1) * ang(Zxx))
    # ypreComplex = unnormalize(exp(y) * exp(complex(0, 1) * ang(Zxx)), abss(Zxx))  # wrong code
    _, xrec = istft(ypreComplex, freq)
    dataWrite = xrec.astype(np.int16)
    wavfile.write(output_path, freq, dataWrite)


def test_model_mfcc(model, input_path, output_path, neighbor, nffts, normal_flag=0):
    _, s = wavfile.read(input_path)
    _, _, Zxx = stft(s, freq)
    Zxx1 = log((abss(Zxx)).T+1e-7)
    y_input = combine_with_mfcc(input_path, neighbor=neighbor, nfft=nffts, normal_flag=normal_flag)
    y = model.predict(y_input)
    y = (np.delete(y, np.s_[-39:], axis=1)).T   # delete mfcc data
    ypreComplex = exp(y) * exp(complex(0, 1) * ang(Zxx)) if normal_flag == 0 \
        else exp(unnormalize(y, Zxx1)) * exp(complex(0, 1) * ang(Zxx))
    # ypreComplex = unnormalize(exp(y) * exp(complex(0, 1) * ang(Zxx)), abss(Zxx))  # wrong code
    _, xrec = istft(ypreComplex, freq)
    dataWrite = xrec.astype(np.int16)
    wavfile.write(output_path, freq, dataWrite)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def test_model_GRU(model, input_path, output_path, neighbor, nffts, normal_flag=0):
    _, s = wavfile.read(input_path)
    _, _, Zxx = stft(s, freq)
    Zxx1 = log((abss(Zxx)).T+1e-7)
    y_input = make_window_buffer(input_path, neighbor=neighbor, nfft=nffts, normal_flag=normal_flag)
    y = model.predict(np.reshape(y_input, [1, -1, 129])).T
    y = np.reshape(y, [129, -1])
    # print(y.shape, unnormalize(y, abss(Zxx)).shape, unnormalize(y, abss(Zxx)).dtype)
    ypreComplex = exp(y) * exp(complex(0, 1) * ang(Zxx)) if normal_flag == 0 \
        else exp(unnormalize(y, Zxx1)) * exp(complex(0, 1) * ang(Zxx))
    # ypreComplex = unnormalize(exp(y) * exp(complex(0, 1) * ang(Zxx)), abss(Zxx))  # wrong code
    _, xrec = istft(ypreComplex, freq)
    dataWrite = xrec.astype(np.int16)
    wavfile.write(output_path, freq, dataWrite)


# these methods below are used in former version, no use now
def list_dir(path, list_name, start=0, end=-1):
    """
    get the list of all paths of every file in 'path' from 'start' to 'end' in order
    :param path: dir path
    :param list_name:
    :param start: start index
    :param end: end index, -1 by default which means the end
    """
    for file in os.listdir(path)[start:end]:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_dir(file_path, list_name, start=start, end=end)
        else:
            list_name.append(file_path)


def getCoupleAbs(xdir, ydir, nfft=256, fr=1):
    """
    get pure and noisy STFT data from xdir and ydir
    :param xdir: pure data dir
    :param ydir: noisy data dir
    :param nfft: fft width
    :param fr: number of frames per frame group
    :return: Zxx, Zyy array
    """
    _, x = wavfile.read(xdir)
    _, y = wavfile.read(ydir)
    _, _, Zxx = stft(x, freq, nfft=nfft)
    _, _, Zyy = stft(y, freq, nfft=nfft)
    if fr == 1:
        return np.abs(Zxx.T), np.abs(Zyy.T)
    else:
        tempX, tempY = np.abs(Zxx.T), np.abs(Zyy.T)
        N_frame = int(np.floor(Zxx.shape[1] / fr))
        return tempX[:int(N_frame * fr), :].reshape((N_frame, int(fr * (nfft/2+1)))),\
            tempY[:int(N_frame * fr), :].reshape((N_frame, int(fr * (nfft/2+1))))


def getCoupleAbsOne(xdir, nfft=256, fr=1):
    """
    get pure and noisy STFT data from xdir
    :param xdir: pure data dir
    :param nfft: fft width
    :param fr: number of frames per frame group
    :return: Zxx array
    """
    _, x = wavfile.read(xdir)
    _, _, Zxx = stft(x, freq, nfft=nfft)
    if fr == 1:
        return np.abs(Zxx.T)
    else:
        tempX = np.abs(Zxx.T)
        N_frame = int(np.floor(Zxx.shape[1] / fr))
        return tempX[:int(N_frame * fr), :].reshape((N_frame, int(fr * (nfft/2+1))))


def add_layer(inputs, in_size, out_size, act_func=None):
    """
    add a layer of TensorFlow
    :param inputs:
    :param in_size:
    :param out_size:
    :param act_func:
    :return:
    """
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if act_func is None:
            return Wx_plus_b
        else:
            return act_func(Wx_plus_b)


class Dense:

    def __init__(self, in_dim, out_dim, func=lambda x: x):
        self.W = tf.Variable(np.random.RandomState().uniform(low=-0.1, high=0.1, size=(in_dim, out_dim))
                             .astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.func = func
        self.params = [self.W, self.b]
        self.ae = Autoencoder(in_dim, out_dim, self.W, self.func)
        self.z = None

    def f_prop(self, x):
        u = tf.matmul(x, self.W) + self.b
        self.z = self.func(u)
        return self.z

    def pretrain(self, x, noise):
        cost, reconst_x = self.ae.reconst_error(x, noise)
        return cost, reconst_x


class Autoencoder:

    def __init__(self, vis_dim, hid_dim, W, func=lambda x: x):
        self.W = W
        self.a = tf.Variable(np.zeros(vis_dim).astype('float32'), name='a')
        self.b = tf.Variable(np.zeros(hid_dim).astype('float32'), name='b')
        self.func = func
        self.params = [self.W, self.a, self.b]

    def encode(self, x):
        u = tf.matmul(x, self.W) + self.b
        return self.func(u)

    def decode(self, x):
        u = tf.matmul(x, tf.transpose(self.W)) + self.a
        return self.func(u)

    def f_prop(self, x):
        y = self.encode(x)
        return self.decode(y)

    def reconst_error(self, x, noise):
        tilde_x = x * noise
        reconst_x = self.f_prop(tilde_x)
        error = tf.reduce_mean(tf.reduce_sum((x - reconst_x)**2, 1))
        return error, reconst_x


def bark2lin(bark):
    # assert np.shape(bark)[1] == 129
    tmp = zer([129])
    tmp[0], tmp[1] = bark[0]/2, bark[0]/2
    tmp[2], tmp[3] = bark[1]/2, bark[1]/2
    tmp[4], tmp[5] = bark[2]/2, bark[2]/2
    tmp[6] = bark[3]
    tmp[7], tmp[8] = bark[4]/2, bark[4]/2
    tmp[9], tmp[10] = bark[5]/2, bark[5]/2
    tmp[11], tmp[12] = bark[6]/2, bark[6]/2
    tmp[13], tmp[14] = bark[7]/2, bark[7]/2
    tmp[15], tmp[16], tmp[17] = bark[8]/3, bark[8]/3, bark[8]/3
    tmp[18], tmp[19], tmp[20] = bark[9]/3, bark[9]/3, bark[9]/3
    tmp[21], tmp[22], tmp[23] = bark[10]/3, bark[10]/3, bark[10]/3
    tmp[24], tmp[25], tmp[26], tmp[27] = bark[11]/4, bark[11]/4, bark[11]/4, bark[11]/4
    tmp[28], tmp[29], tmp[30], tmp[31], tmp[32] = bark[12]/5, bark[12]/5, bark[12]/5, bark[12]/5, bark[12]/5
    tmp[33], tmp[34], tmp[35], tmp[36], tmp[37] = bark[13]/5, bark[13]/5, bark[13]/5, bark[13]/5, bark[13]/5
    tmp[38], tmp[39], tmp[40], tmp[41], tmp[42], tmp[43] = \
        bark[14]/6, bark[14]/6, bark[14]/6, bark[14]/6, bark[14]/6, bark[14]/6

    tmp[44], tmp[45], tmp[46], tmp[47], tmp[48], tmp[49], tmp[50] \
        = bark[15]/7, bark[15]/7, bark[15]/7, bark[15]/7, bark[15]/7, bark[15]/7, bark[15]/7

    tmp[51], tmp[52], tmp[53], tmp[54], tmp[55], tmp[56], tmp[57], tmp[58], tmp[59] \
        = bark[16]/9, bark[16]/9, bark[16]/9, bark[16]/9, bark[16]/9, bark[16]/9, bark[16]/9, bark[16]/9, bark[16]/9

    tmp[60], tmp[61], tmp[62], tmp[63], tmp[64], tmp[65], tmp[66], tmp[67], tmp[68], tmp[69], tmp[70] \
        = \
        bark[17]/11, bark[17]/11, bark[17]/11, bark[17]/11, bark[17]/11, bark[17]/11, bark[17]/11, bark[17]/11, \
        bark[17]/11, bark[17]/11, bark[17]/11

    tmp[71], tmp[72], tmp[73], tmp[74], tmp[75], tmp[76], tmp[77], tmp[78], tmp[79], tmp[80], tmp[81], tmp[82], \
        tmp[83], tmp[84] \
        = \
        bark[18]/14, bark[18]/14, bark[18]/14, bark[18]/14, bark[18]/14, bark[18]/14, bark[18]/14, bark[18]/14, \
        bark[18]/14, bark[18]/14, bark[18]/14, bark[18]/14, bark[18]/14, bark[18]/14

    tmp[85], tmp[86], tmp[87], tmp[88], tmp[89], tmp[90], tmp[91], tmp[92], tmp[93], tmp[94], tmp[95], tmp[96], \
        tmp[97], tmp[98], tmp[99], tmp[100], tmp[101], tmp[102] \
        = \
        bark[19]/18, bark[19]/18, bark[19]/18, bark[19]/18, bark[19]/18, bark[19]/18, bark[19]/18, bark[19]/18, \
        bark[19]/18, bark[19]/18, bark[19]/18, bark[19]/18, bark[19]/18, bark[19]/18, bark[19]/18, bark[19]/18, \
        bark[19]/18, bark[19]/18

    tmp[103], tmp[104], tmp[105], tmp[106], tmp[107], tmp[108], tmp[109], tmp[110], tmp[111], tmp[112], tmp[113], \
        tmp[114], tmp[115], tmp[116], tmp[117], tmp[118], tmp[119], tmp[120], tmp[121], tmp[122], tmp[123] \
        = \
        bark[20]/21, bark[20]/21, bark[20]/21, bark[20]/21, bark[20]/21, bark[20]/21, bark[20]/21, bark[20]/21, \
        bark[20]/21, bark[20]/21, bark[20]/21, bark[20]/21, bark[20]/21, bark[20]/21, bark[20]/21, bark[20]/21, \
        bark[20]/21, bark[20]/21, bark[20]/21, bark[20]/21, bark[20]/21

    tmp[124], tmp[125], tmp[126], tmp[127], tmp[128] = bark[21]/5, bark[21]/5, bark[21]/5, bark[21]/5, bark[21]/5
    return tmp


def fr2bark(Zyy):
    """
    shape([129, ?]) --> shape([?, 22])
    :param Zyy:
    :return:
    """
    tmp = zer([Zyy.shape[1], 22])
    zt = Zyy.T
    for i in range(zt.shape[0]):
        tmp[i][0] = np.mean(zt[i][0:2])
        tmp[i][1] = np.mean(zt[i][2:4])
        tmp[i][2] = np.mean(zt[i][4:6])
        tmp[i][3] = zt[i][6]
        tmp[i][4] = np.mean(zt[i][7:9])
        tmp[i][5] = np.mean(zt[i][9:11])
        tmp[i][6] = np.mean(zt[i][11:13])
        tmp[i][7] = np.mean(zt[i][13:15])
        tmp[i][8] = np.mean(zt[i][15:18])
        tmp[i][9] = np.mean(zt[i][18:21])
        tmp[i][10] = np.mean(zt[i][21:24])
        tmp[i][11] = np.mean(zt[i][24:28])
        tmp[i][12] = np.mean(zt[i][28:33])
        tmp[i][13] = np.mean(zt[i][33:38])
        tmp[i][14] = np.mean(zt[i][38:44])
        tmp[i][15] = np.mean(zt[i][44:51])
        tmp[i][16] = np.mean(zt[i][51:60])
        tmp[i][17] = np.mean(zt[i][60:71])
        tmp[i][18] = np.mean(zt[i][71:85])
        tmp[i][19] = np.mean(zt[i][85:103])
        tmp[i][20] = np.mean(zt[i][103:124])
        tmp[i][21] = np.mean(zt[i][124:])
    return tmp


def bark_rescale(bark):
    """
    rescale bark to [0, 1]
    :param bark:
    :return:
    """
    tmp = np.zeros([np.shape(bark)[0], np.shape(bark)[1]])
    for j in range(np.shape(bark)[0]):
        _sum = np.sum(bark[j]) + 1e-7
        for i in range(np.shape(bark)[1]):
            tmp[j][i] = bark[j][i] / _sum
    return tmp


def bark_dct(bark_rescaled):
    tmp = zer([np.shape(bark_rescaled)[0], np.shape(bark_rescaled)[1]])
    for j in range(np.shape(bark_rescaled)[0]):
        tmp[j] = cv2.dct(bark_rescaled[j]).ravel()
    return tmp


def pack_GRU(xdir):
    _, x = wavfile.read(xdir)
    _, _, Zxx = stft(x, freq)
    Zxx = log((abss(Zxx)) + 1e-7)
    return bark_dct(bark_rescale(fr2bark(Zxx)))


def unpack_GRU(pack):
    pack = np.reshape(pack, [-1, 22])
    print(pack.shape)
    tmp1 = zer([np.shape(pack)[0], np.shape(pack)[1]])
    for j in range(np.shape(pack)[0]):
        tmp1[j] = cv2.idct(pack[j]).ravel()
    tmp2 = zer([np.shape(pack)[0], 129])
    for j in range(np.shape(tmp1)[0]):
        tmp2[j] = bark2lin(tmp1[j])
    return tmp2


def test_GRU(model, input_path, output_path):
    _, s = wavfile.read(input_path)
    _, _, Zxx = stft(s, freq)
    Zxx1 = log((abss(Zxx)).T + 1e-7)
    print(Zxx1.shape)
    yt = pack_GRU(input_path)
    y = model.predict(np.reshape(yt, [1, -1, 22]))
    y = unpack_GRU(y)
    print(y.shape)
    ypreComplex = exp(y.T * Zxx1.T) * exp(complex(0, 1) * ang(Zxx))
    _, xrec = istft(ypreComplex, freq)
    dataWrite = xrec.astype(np.int16)
    wavfile.write(output_path, freq, dataWrite)
