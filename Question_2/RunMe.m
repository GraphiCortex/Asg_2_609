clear all;

global eNa eK eL ...
    gNa gK gL gCaL ...
    tauNbar tauH Ca_ex RTF ...
    thetaM sigmaM thetaN sigmaN thetaS sigmaS ...
    f iapp eps kca C ...
    from to
    
% Nernest potentials in mV
eNa=50; 
eK=-90; 
eL=-70; 

% Conductances in nS
gNa=450;
gK=50;
gL=2;
gCaL=20;

% Applied current parameters
iapp=100;

from=200;
to=600;

% Time constants in mS
tauNbar=10;
tauH=1;

% Half-activation constants in mV
thetaM=-35;
thetaN=-30;
sigmaM=-5;
sigmaN=-5;
thetaS=-20;
sigmaS=-0.05;


% Capacitance in pF
C=100;

% Other constants
RTF=26.7;
Ca_ex=2.5;
f=0.1;
eps=0.0015;
kca=0.3;


% Initial Conditons
v_i=-72.14; 
n_i=.0002;
ca_i=0.103;
h_i=0.01;

init=[n_i;h_i;ca_i;v_i];

% Duration of the simulation   
time=0:.01:800;
    
% Run the model via the ode113 solver
[time,output] = ode113('DiffEquations',time,init);
voltage = output(:,4);


figure();
plot(time, voltage, 'k-', 'Linewidth', 2);
ylabel('Voltage (mV)');
xlabel('Time (ms)');

box off 

