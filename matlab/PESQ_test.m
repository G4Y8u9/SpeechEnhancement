% [MOS_Raw, MOS_LQO]=pesq('01.wav', 'output_01.wav', '+16000');

SNR_ = 10;
pure_path = strcat('G:\trainData\-5db\test\pure\');
noisy_path = strcat('G:\trainData\-5db\test\noisy\');
output_path = strcat('G:\trainData\-5db\test\output\mfcc\');

fileExt = '*.wav';
pure_files = dir(fullfile(pure_path, fileExt));
noisy_files = dir(fullfile(noisy_path, fileExt));
output_files = dir(fullfile(output_path, fileExt));
MOS_noisy = zeros([100,2]);
MOS_output = zeros([100,2]);
for i = 1:100
    [MOS_noisy(i), MOS_noisy(i+100)] = pesq(...
        strcat(pure_path, '\', pure_files(i).name), ...
        strcat(noisy_path, '\', noisy_files(i).name), '+16000');
    [MOS_output(i), MOS_output(i+100)] = pesq(...
        strcat(pure_path, '\', pure_files(i).name), ...
        strcat(output_path, '\', output_files(i).name), '+16000');
    disp(['evaluating file no.', num2str(i)]);
end
disp([mean(MOS_noisy(101:200)), mean(MOS_output(101:200))]);
