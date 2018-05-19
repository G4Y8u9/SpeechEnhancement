purepath = 'E:\trainData\data\ChineseTrain\5db\pure';
noisepath = 'E:\trainData\data1\Noise_16k';
fileExt = '*.wav';
wavFiles = dir(fullfile(purepath, fileExt));
noisyFiles = dir(fullfile(noisepath, fileExt));
N=1000;

wavFiles_N = getranditem(wavFiles, N);
savepath_pure = 'E:\trainData\newdata\5db\pure\';
savepath_noisy = 'E:\trainData\newdata\5db\noisy\';

lenn = size(noisyFiles, 1);
for j = 1:lenn
    display1 = [noisyFiles(j, 1).name, ' process...'];
    disp(display1);
    noisyDir = strcat(noisepath,'\', noisyFiles(j, 1).name);
    [noise, Fs] = audioread(noisyDir);
    wavFiles_N = getranditem(wavFiles, N);
    for i = 1:N
        pureDir = strcat(purepath, '\', wavFiles_N(i, 1).name);
        [pure, Fs] = audioread(pureDir);
        noisywav = addnoise(pure, subset(noise, size(pure,1)), 5);
        filename_pure = strcat(savepath_pure, '\',...
            noisyFiles(j, 1).name(1:end-4), ...
            '_', wavFiles_N(i, 1).name);
        filename_noisy = strcat(savepath_noisy, '\',...
            noisyFiles(j, 1).name(1:end-4), ...
            '_', wavFiles_N(i, 1).name);
        audiowrite(filename_pure, pure, 16000);
        audiowrite(filename_noisy, noisywav, 16000);
        display3 = ['Noise No.', num2str(j),...
            ', pure No.', num2str(i), ' processed!'];
        disp(display3);
    end
    display2 = [noisyFiles(j, 1).name, ' processed!'];
    disp(display2);
end