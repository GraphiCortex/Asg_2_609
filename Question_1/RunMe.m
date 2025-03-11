clear all;
clc;

global vNa vK vL gNa gK gL C Iapp from to 

from = 150;
to = 350;

C=1;

gNa = 120;  
vNa = 50;
gK = 36;  
vK = -77;
gL = 0.3;  
vL = -54.4;
Iapp = 20;
  
% Integrate the model

tspan = 0:0.01:500;

v_i=-65; 
m_i=0.1;
h_i=0.1;
n_i=0.1;

init_cond = [v_i, m_i, h_i, n_i];

[t, sol] = ode45(@DiffEquations, tspan, init_cond);
modelTrace = sol(:,1);   


fig = figure();

plot(t, modelTrace, 'k-', 'Linewidth', 2);
hold on

xlabel('Time (ms)');
ylabel('Voltage (mV)');