% This matlab program executes parallel EKF and EKS, where maternal ECG and
% fetal ECG, mixed in a single observation, are modelled simultaneously.
% This models fully overlapping several ECGs. 
% DaISy database has been used , which consists a single dataset of
% cutaneous potential recording of a pregnant woman. A total of 8 channels
% (5 abdominal and 3 thoaracic) are available, sampled at 250Hz and lasting
% 10 seconds. 

%%
clc
clear all
close all;

%%
load foetal_ecg.mat;                              % Original Recorded Signal with 5 abdominal channels and 3 thoracic channels              
load RealData_OptimumParams;                      % Estimated Model Parameters
load peaks_mother.mat;                            % Detected Peaks of Maternal ECG
load peaks_fetus.mat;                             % Detected Peaks of Foetal ECG

%%
data = foetal_ecg(2,:);                           % Recorded Data from the second channel being used
fs = 250;                                         % Sampling Frequency

t = (0:length(data)-1)/fs;                        % Calculating time intervals using sampling frequency

bsline = LPFilter(data,.7/fs);                    % Returns second order zero phase low pass filter, inputs data vector, cut off frequency

x=data-bsline;                                    % Baseline wander removed data vector

%% Modelling Maternal ECG

peaks1=peaks_mother;                              % Vector of R peaks pulse train of maternal ECG
I=find(peaks1);                                   % Finding indices of maternal peaks

[phase1 phasepos1] = PhaseCalculation(peaks1);    % ECG phase calculation for maternalgiven R peaks train

teta = 0;                                         % Desired phase shift, teta>0 corresponds phase leads and teta<0 corresponds phase lags
pphase1 = PhaseShifting(phase1,teta);             % Phase Shifting- inputs calculated ECG phase and desired phase shift, output the shifted phase

dif_I = zeros(length(I)-1,1);
for i=2:length(I)
dif_I(i-1)=I(i)-I(i-1);
end
bins1 = round(mean(dif_I));                       % Number of phase bins

% Calculation of mean and SD of maternal ECG waveforms in different beats, inputs
% ECG data vector, Shifted phase, number of bins, flag 1 is used to align
% baseline to zero, flag 0 means no zero baseline alignment
[ECGmean1,ECGsd1,meanphase1] = MeanECGExtraction(x,pphase1,bins1,1); 

%%
N1 = length(OptimumParams1)/3;                    % Number of Gaussian kernels
JJ1 = find(peaks1);
fm1 = fs./diff(JJ1);                              % Heart-rate
w1 = mean(2*pi*fm1);                              % Average heart-rate in rads.
wsd1 = std(2*pi*fm1,1);                           % Heart-rate standard deviation in rads.


%% Modelling Fetal ECG

peaks2=peaks_fetus;                               % Vector of R peaks pulse train of fetal ECG
I=find(peaks1);                                   % Finding indices of fetal peaks

[phase2 phasepos2] = PhaseCalculation(peaks2);    % ECG phase calculation for fetal R peaks train

teta = 0;                                         % Desired phase shift, teta>0 corresponds phase leads and teta<0 corresponds phase lags
pphase2 = PhaseShifting(phase2,teta);             % Phase Shifting- inputs calculated ECG phase and desired phase shift, output the shifted phase

dif_I = zeros(length(I)-1,1);
for i=2:length(I)
dif_I(i-1)=I(i)-I(i-1);
end
bins2 = round(mean(dif_I));                       % Number of phase bins

% Calculation of mean and SD of fetal ECG waveforms in different beats, inputs
% ECG data vector, Shifted phase, number of bins, flag 1 is used to align
% baseline to zero, flag 0 means no zero baseline alignment
[ECGmean2,ECGsd2,meanphase2] = MeanECGExtraction(x,pphase2,bins2,1); 

%%
N2 = length(OptimumParams2)/3;                    % Number of Gaussian kernels
JJ2 = find(peaks2);
fm2 = fs./diff(JJ2);                              % Heart-rate
w2 = mean(2*pi*fm2);                              % Average heart-rate in rads.
wsd2 = std(2*pi*fm2,1);                           % Heart-rate standard deviation in rads.


%%

y = [phase1; phase2 ; x];                         % Matrix of Observation Signals- phase observations for both maternal and fetal ECG and BW removed data vector
X0 = [-pi -pi 0 0]';                              % Initial state vector

P0 = [(2*pi)^2 0 0 0;0 (2*pi)^2 0 0;0 0 (10*max(abs(x))).^2 0; 0 0 0 (10*max(abs(x))).^2 ]; % Covariance matrix of the initial state vector

%Q-Covariance matrix of the process noise vector
Q = diag( [ (.1*OptimumParams1(1:N1)).^2 (.05*ones(1,N1)).^2 (.05*ones(1,N1)).^2 (wsd1)^2 ,...
    (.1*OptimumParams2(1:N2)).^2 (.05*ones(1,N2)).^2 (.05*ones(1,N2)).^2 (wsd2)^2 ,...
    (.05*mean(ECGsd1(1:round(length(ECGsd1)/10))))^2, (.05*mean(ECGsd2(1:round(length(ECGsd2)/10))))^2] );

%R- Covariance matrix of the observation noise vector
R = [(w1/fs).^2/12 0 0 0;0 (w2/fs).^2/12 0 0;...
    0  0 (mean(ECGsd1(1:round(length(ECGsd1)/10)))).^2 0; 0 0 0 (1*mean(ECGsd2(1:round(length(ECGsd2)/10)))).^2];

Wmean = [OptimumParams1 OptimumParams2 w1 w2 0 0]';    % Mean Process noise vector  
Vmean = [0 0 0]';                                      % Mean Observation noise vector
Inits = [OptimumParams1 OptimumParams2 w1 w2 fs fs];   % Filter Initialization Parameters

InnovWinLen = ceil(.5*fs);                       % Innovations monitoring window length
tau = [];                                        % Kalman filter forgetting time. tau=[] for no forgetting factor
gamma = 1;                                       % Observation covariance adaptation-rate. 0<gamma<1 and gamma=1 for no adaptation
AdaptWinLen = ceil(fs/2);                        % Window length for observation covariance adaptation


%EKSmoother function provides EKF and EKS for noisy ECG data.Outputs are EKF of noisy
%data, EKF state vector covariance matrix, EKS of noisy data, EKS state
%vector covariance matrix and measure of innovations signal whiteness
[Xekf,Pekf,Xeks,Peks,ak1,ak2] = EKSmoother2ECG(y,X0,P0,Q,R,Wmean,Vmean,Inits,InnovWinLen,tau,gamma,AdaptWinLen,1);

%% 
% Xekf has four columns, first two columns phase estimates, next two
% columns denoised maternal ECG and denoised fetal ECG
Xekf1 = Xekf(3,:);
Xeks1 = Xeks(3,:);
Xekf2 = Xekf(4,:);
Xeks2 = Xeks(4,:);

% Baseline Wander Removal for EKF and EKS outputs of Maternal and Fetal ECG
bsline = LPFilter(Xekf1,.7/fs);
Xekf1 = Xekf1 - bsline;
Xekf1 = Xekf1 - mean(Xekf1);

bsline = LPFilter(Xekf2,.7/fs);
Xekf2 = Xekf2 - bsline;
Xekf2 = Xekf2 - mean(Xekf2);

bsline = LPFilter(Xeks1,.7/fs);
Xeks1 = Xeks1 - bsline;
Xeks1 = Xeks1 - mean(Xeks1);

bsline = LPFilter(Xeks2,.7/fs);
Xeks2 = Xeks2 - bsline;
Xeks2 = Xeks2 - mean(Xeks2);


%% % Plotting noisy ECG data and denoised ECG
figure
plot(t,x,t,Xekf1,t, Xeks1);
%hold on;
grid;
title('Maternal ECG Estimation using Parallel EKF and Parallel EKS','FontWeight','bold'); xlabel('Time');ylabel('Relative Amplitude');
legend('Original ECG','EKF Output of Maternal ECG','EKS Output of Maternal ECG');

figure
plot(t,x,t, Xekf2, t, Xeks2);legend('EKS estimation of maternal ECGs');
grid;
title('Fetal ECG Estimation using Parallel EKF and Parallel EKS','FontWeight','bold'); xlabel('Time');ylabel('Relative Amplitude');
legend('Original ECG','EKF Output of Fetal ECG','EKS Output of Fetal ECG');

