function [MOS_Raw, MOS_LQO]=pesq(CleanSignal,DegradedSignal,rate)
%all should be in str
[~,strout]=system(['PESQ ',rate,' ',CleanSignal,' ',DegradedSignal]);
c=strfind(strout,'(Raw MOS, MOS-LQO):');
if isempty(c)
    MOS_Raw='NULL';
    MOS_LQO='NULL';
else
    MOS_Raw=str2double(strout(c+23:c+28));
    MOS_LQO=str2double(strout(c+29:end-1));
end