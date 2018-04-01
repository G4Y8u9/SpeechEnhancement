"""
This py file is to proove that human's ears are not sensible to the phase of sound
So, we can directly use the phase of noisy sound's stft data
then train the abs of the data with DNN
last convert it into a denoised wav
"""
from scipy.io import wavfile
from scipy.signal import stft, istft
import numpy as np


# read pure data
fs, data = wavfile.read('./PureSound.wav')
# read noisy data
_, noisedata = wavfile.read('./NoisySound.wav')
# stft both data
_, _, Xdata = stft(data, fs)
_, _, Xnoisedata = stft(noisedata, fs)
# istft, with abs of X, combine with phase(angle) of Xnoise
_, newData = istft(np.abs(Xdata) * np.exp(complex(0, 1) * np.angle(Xnoisedata)), fs)
# turn into int16 type s.t. write as wav file
dataWrite = newData.astype(np.int16)
# write as wav file
wavfile.write('./PureSoundWithNoisyPhase.wav', 16000, dataWrite)
