% a = '123.wav';
% b = a(1:end-4);
wavsong = audioread('C:\Users\G4Y8u9\Desktop\yumengji.wav');
song16k = resample(wavsong, 1, 3);
songsingle = sum(song16k, 2) / size(song16k, 2);
audiowrite('C:\Users\G4Y8u9\Desktop\yumengji16ksingle.wav', songsingle, 16000);

% path = 'E:\trainData\data1\QutNoise';
% path_new = 'E:\trainData\data1\Noise_16k\';
% fileExt = '*.wav';
% wavFiles = dir(fullfile(path, fileExt));
% lenw = size(wavFiles, 1);
% for j = 1:lenw
%     wavDir = strcat(path,'\', wavFiles(j, 1).name);
%     display1 = [wavDir, ' processing...'];
%     disp(display1);
%     [noise_origin, Fs] = audioread(wavDir);
%     noise_16k = resample(noise_origin, 1, 3);
%     noise_mono = sum(noise_16k, 2)/size(noise_16k, 2);
%     filename_pure = strcat(path_new, wavFiles(j, 1).name);
%     audiowrite(filename_pure, noise_mono, 16000);
%     display2 = [wavDir, ' processed.'];
%     disp(display2);
% end;