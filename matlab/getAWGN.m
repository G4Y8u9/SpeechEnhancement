function getAWGN(x, SNR_dB, fileName, Fs)
    y = awgn(x, SNR_dB, 'measured');
    audiowrite(fileName, y./max(abs(y)), Fs)
end