%% 
% This matlab program executes parallel EKF and EKS, where maternal ECG and
% fetal ECG, mixed in a single observation, that are modelled simultaneously.
% This models fully overlapping several ECGs. 
% PhysioNet Non-Invasive Fetal Electrocardiogram database has been used ,
% which consists a series of 55 multichannel abdominal fECG recordings,
% taken from a single subject between 21 to 40 weeks of pregnancy. The
% recordings include 2 thoracic signals and 3 or 4 abdominal signals. The
% signals were recorded at 1kHz,16-bit resolution with a bandpass filter
% (0.01Hz-100Hz) and a main notch filter (50Hz).

%%
clc
clear all
close all;

%%
load ecgca771_edfm.mat;                            % Original signal of ecgca771 of the PhysioNet Database
data = val(3,:);                                   % Selecting channel 3
fs = 1000;                                         % Sampling Frequency 1000 Hz
f = 1.35;                                          % Parameter for Peak Detection
data=data/100;                                     % Making relative amplitude dividing by 100
t = (0:length(data)-1)/fs;                         % Calculating timing values

bsline = LPFilter(data,.7/fs);                     % Baseline calculation using LP Filter

x = data - bsline;                                 % Baseline wander removal

%% Modelling Maternal ECG
peaks1 = PeakDetection(x,f/fs);                    % Peak Detection
I = find(peaks1);

[phase1 phasepos1] = PhaseCalculation(peaks1);     % Phase Calculation

teta = 0;                                          % Desired phase shift
pphase1 = PhaseShifting(phase1,teta);              % Phase shifting

dif_I = zeros(length(I)-1,1);                      % Calculating number of phase bins
for i=2:length(I)
dif_I(i-1)=I(i)-I(i-1);
end
bins1 = round(mean(dif_I));
[ECGmean1,ECGsd1,meanphase1] = MeanECGExtraction(x,pphase1,bins1,1); % Mean ECG extraction


ECGBeatFitter(ECGmean1,ECGsd1,meanphase1,'OptimumParams');           % ECG beat fitter GUI
OptimumParams1 = OptimumParams;                                      % Assigning Optimal Parameters

N1 = length(OptimumParams1)/3;                                       % Number of Gaussian kernels
JJ1 = find(peaks1);
fm1 = fs./diff(JJ1);                                                 % Heart-rate
w1 = mean(2*pi*fm1);                                                 % Average heart-rate in rads.
wsd1 = std(2*pi*fm1,1);                                              % heart-rate standard deviation in rads.

%% Modelling Fetal ECG

peaks2 = PeakDetection(x,f/fs);                                      % Peak Detection
I= find(peaks2);

[phase2 phasepos2] = PhaseCalculation(peaks2);                       % Phase Calculation

teta = 0;                                                            % Desired phase shift
pphase2 = PhaseShifting(phase2,teta);                                % Phase shifting

dif_I = zeros(length(I)-1,1);                                        % Calculating number of phase bins
for i=2:length(I)
dif_I(i-1)=I(i)-I(i-1);
end
bins2 = round(mean(dif_I));

[ECGmean2,ECGsd2,meanphase2] = MeanECGExtraction(x,pphase2,bins2,1); % Mean ECG extraction

ECGBeatFitter(ECGmean2,ECGsd2,meanphase2,'OptimumParams');           % ECG beat fitter GUI
OptimumParams2 = OptimumParams;                                      % Assigning Optimal Parameters

N2 = length(OptimumParams2)/3;                                       % Number of Gaussian kernels
JJ2 = find(peaks2);
fm2 = fs./diff(JJ2);                                                 % Heart-rate
w2 = mean(2*pi*fm2);                                                 % Average heart-rate in rads.
wsd2 = std(2*pi*fm2,1);                                              % Heart-rate standard deviation in rads.

%%

y = [phase1; phase2; x];           % Matrix of Observation Signals- phase observations for both maternal and fetal ECG and BW removed data vector

X0 = [-pi -pi 0 0]';               % Initial state vector

P0 = [(2*pi)^2 0 0 0;0 (2*pi)^2 0 0;0 0 (10*max(abs(x))).^2 0; 0 0 0 (10*max(abs(x))).^2 ]; % Covariance matrix of the initial state vector

%Q-Covariance matrix of the process noise vector
Q = diag( [ (.1*OptimumParams1(1:N1)).^2 (.05*ones(1,N1)).^2 (.05*ones(1,N1)).^2 (wsd1)^2 ,...
    (.1*OptimumParams2(1:N2)).^2 (.05*ones(1,N2)).^2 (.05*ones(1,N2)).^2 (wsd2)^2 ,...
    (.05*mean(ECGsd1(1:round(length(ECGsd1)/10))))^2, (.05*mean(ECGsd2(1:round(length(ECGsd2)/10))))^2] );

%R- Covariance matrix of the observation noise vector
R = [(w1/fs).^2/12 0 0 0;0 (w2/fs).^2/12 0 0;...
    0  0 (mean(ECGsd1(1:round(length(ECGsd1)/10)))).^2 0; 0 0 0 (1*mean(ECGsd2(1:round(length(ECGsd2)/10)))).^2];

Wmean = [OptimumParams1 OptimumParams2 w1 w2 0 0]';             % Mean Process noise vector
Vmean = [0 0 0]';                                               % Mean Observation noise vector
Inits = [OptimumParams1 OptimumParams2 w1 w2 fs fs];            % Filter Initialization Parameters

InovWlen = ceil(.5*fs);     % Innovations monitoring window length
tau = [];                   % Kalman filter forgetting time. tau=[] for no forgetting factor
gamma = 1;                  % Observation covariance adaptation-rate. 0<gamma<1 and gamma=1 for no adaptation
RadaptWlen = ceil(fs/2);    % Window length for observation covariance adaptation

%EKSmoother function provides EKF and EKS for noisy ECG data.Outputs are EKF of noisy
%data, EKF state vector covariance matrix, EKS of noisy data, EKS state
%vector covariance matrix and measure of innovations signal whiteness
[Xekf,Phat,Xeks,PSmoothed,ak1,ak2] = EKSmoother2ECG(y,X0,P0,Q,R,Wmean,Vmean,Inits,InovWlen,tau,gamma,RadaptWlen,1);

%%
% Xekf has four columns, first two columns phase estimates, next two
% columns denoised maternal ECG and denoised fetal ECG
Xekf1 = Xekf(3,:);
Phat1 = squeeze(Phat(3,3,:))';
Xeks1 = Xeks(3,:);
PSmoothed1 = squeeze(PSmoothed(3,3,:))';

Xekf2 = Xekf(4,:);
Phat2 = squeeze(Phat(4,4,:))';
Xeks2 = Xeks(4,:);
PSmoothed2 = squeeze(PSmoothed(4,4,:))';

% Baseline Wander Removal for EKF and EKS outputs of Maternal and Fetal ECG
bsline = LPFilter(Xekf1,.7/fs);Xekf1 = Xekf1 - bsline;Xekf1 = Xekf1 - mean(Xekf1);

bsline = LPFilter(Xekf2,.7/fs);Xekf2 = Xekf2 - bsline;Xekf2 = Xekf2 - mean(Xekf2);

bsline = LPFilter(Xeks1,.7/fs);Xeks1 = Xeks1 - bsline;Xeks1 = Xeks1 - mean(Xeks1);

bsline = LPFilter(Xeks2,.7/fs);Xeks2 = Xeks2 - bsline;Xeks2 = Xeks2 - mean(Xeks2);

%% %% % Plotting original ECG and par-EKS maternal and fetal estimations

figure 
subplot(3,1,1); plot(t,data); grid off; title('Recorded Signal (Dataset ecgca274 channel 5)');
subplot(3,1,2); plot(t,Xeks1); grid off; title('par-EKS maternal estimation');
subplot(3,1,3);plot(t, Xeks2);grid off;title('par-EKS fetal estimation');xlabel('time(s)');
ylabel('Relative Amplitude');

