% This matlab program executes sequential EKF and EKS, where one ECG is extracted according to a deflation procedure. 
% The first step extracts maternal ECG from the raw recording. After
% substracting maternal ECG from from the original signal, the next step is
% extraction of fECG from the residual signal.This program extracts
% fetal ECG from maternal ECG extracted Recordings. 
% DaISy database has been used , which consists a single dataset of
% cutaneous potential recording of a pregnant woman. A total of 8 channels
% (5 abdominal and 3 thoaracic) are available, sampled at 250Hz and lasting
% 10 seconds. 

%%
clc
clear all;
close all;

%%
load foetal_ecg.mat;                              % Original Recorded Signal with 5 abdominal channels and 3 thoracic channels   
load substracted_channel1.mat;                    % Maternal ECG Substracted Data Set              
load RealData_OptimumParams;                      % Estimated Model Parameters
load peaks_fetus.mat;                             % Detected Peaks of Foetal ECG

%%
data = foetal_ecg(2,:);                           % Recorded Data from the second channel being used
fs = 250;                                         % Sampling Frequency
% 
t = (0:length(data)-1)/fs;                        % Calculating time intervals using sampling frequency
% 
bsline = LPFilter(data,.7/fs);                    % Returns second order zero phase low pass filter, inputs data vector, cut off frequency
% 
x1=data-bsline;                                   % Baseline wander removed data vector

x= substracted_channel1;                              % Maternal ECG Substracted Data Set being considered
%%
peaks=peaks_fetus;                                % Vector of R peaks pulse train of fetal ECG
I=find(peaks);                                    % Finding indices of fetal peaks

[phase phasepos] = PhaseCalculation(peaks);       % ECG phase calculation for fetal R peaks train

teta = 0;                                         % Desired phase shift, teta>0 corresponds phase leads and teta<0 corresponds phase lags
pphase = PhaseShifting(phase,teta);               % Phase Shifting- inputs calculated ECG phase and desired phase shift, output the shifted phase

dif_I = zeros(length(I)-1,1);                     % Calculating number of bins
for i=2:length(I)
dif_I(i-1)=I(i)-I(i-1);
end
bins = round(mean(dif_I));                                  

% Calculation of mean and SD of ECG waveforms in different beats, inputs
% ECG data vector, Shifted phase, number of bins, flag 1 is used to align
% baseline to zero, flag 0 means no zero baseline alignment
[ECGmean,ECGsd,meanphase] = MeanECGExtraction(x,pphase,bins,1); 


%%
N = length(OptimumParams2)/3;                    % Number of Gaussian kernels
fm = fs./diff(I);                                % Heart-rate
w = mean(2*pi*fm);                               % Average heart-rate in rads.
wsd = std(2*pi*fm,1);                            % Heart-rate standard deviation in rads.

%%
y = [pphase;x];                                  % Matrix of Observation Signals- phase observations and BW removed data vector
X0 = [-pi 0]';                                   % Initial state vector
P0 = [(2*pi)^2 0 ;0 (10*max(abs(x))).^2];        % Covariance matrix of the initial state vector

%Q-Covariance matrix of the process noise vector
Q = diag( [ (.1*OptimumParams2(1:N)).^2 (.05*ones(1,N)).^2 (.05*ones(1,N)).^2 (wsd)^2 , (.05*mean(ECGsd(1:round(length(ECGsd)/10))))^2] );

%R- Covariance matrix of the observation noise vector
R = [(w/fs).^2/12 0 ;0 (mean(ECGsd(1:round(length(ECGsd)/10)))).^2];

Wmean = [OptimumParams2 w 0]';                   % Mean Process noise vector
Vmean = [0 0]';                                  % Mean Observation noise vector
Inits = [OptimumParams2 w fs];                   % Filter Initialization Parameters

InnovWinLen = ceil(.5*fs);                       % Innovations monitoring window length
tau = [];                                        % Kalman filter forgetting time. tau=[] for no forgetting factor
gamma = 1;                                       % Observation covariance adaptation-rate. 0<gamma<1 and gamma=1 for no adaptation
AdaptWinLen = ceil(fs/2);                        % Window length for observation covariance adaptation


%EKSmoother function provides EKF and EKS for noisy ECG data. Flag 1 is
%is used to display waitbar. But it is optional. Outputs are EKF of noisy
%data, EKF state vector covariance matrix, EKS of noisy data, EKS state
%vector covariance matrix and measure of innovations signal whiteness
[Xekf,Pekf,Xeks,Peks,a] = EKSmoother(y,X0,P0,Q,R,Wmean,Vmean,Inits,InnovWinLen,tau,gamma,AdaptWinLen,1);

%%
Xekf = Xekf(2,:);                                % Xekf has two columns, first column phase estimates, second column denoised ECG
Xeks = Xeks(2,:);                                % Xeks has two columns, first column phase estimates, second column denoised ECG

bsline = LPFilter(Xekf,.7/fs);
Xekf = Xekf - bsline;
Xekf = Xekf - mean(Xekf);

bsline = LPFilter(Xeks,.7/fs);
Xeks = Xeks - bsline;
Xeks = Xeks - mean(Xeks);

figure                                           % Plotting noisy ECG data and denoised ECG
plot(t,x1,t,Xekf,t,Xeks);
grid;
title('Fetal ECG Estimation using Sequential EKF and Sequential EKS','FontWeight','bold'); xlabel('Time');ylabel('Relative Amplitude');
legend('Original ECG','EKF Output','EKS Output');
