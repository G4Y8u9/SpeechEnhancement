path = 'E:\trainData\data1\Chinese\train';
train_pure = 'E:\SpeechEnhancement\data\ChineseTest\10db\pure\';
train_noisy = 'E:\SpeechEnhancement\data\ChineseTest\10db\noisy\';
dirExt = '';
fileExt = '*.wav';
files = dir(fullfile(path, dirExt));
len = size(files, 1);
for i = 1:len
    fileDir = strcat(path, files(i, 1).name);
    wavFiles = dir(fullfile(fileDir, fileExt));
    lenw = size(wavFiles, 1);
    for j = 1:lenw
        wavDir = strcat(fileDir,'\', wavFiles(j, 1).name);
        [y, Fs] = audioread(wavDir);
        filename_noisy = strcat(train_noisy, wavFiles(j, 1).name);
        filename_pure = strcat(train_pure, wavFiles(j, 1).name);
        getAWGN(y, 10, filename_noisy, Fs);
        audiowrite(filename_pure, y, Fs);
    end;
    display = [fileDir, ' processed'];
    disp(display);
end;

