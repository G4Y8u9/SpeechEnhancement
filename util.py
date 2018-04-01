import tensorflow as tf
import numpy as np
import os
from scipy.io import wavfile
from scipy.signal import stft, istft
from numpy import log, exp, abs as abss, angle as ang, zeros as zer
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
