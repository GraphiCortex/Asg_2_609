function [output] = DiffEquations(t, sol)

global vNa vK vL gNa gK gL C Iapp from to

v=sol(1);
m=sol(2);
h=sol(3);
n=sol(4);

% Na+ Equations and Currents
alphaM = 0.1*((v+40)/(1-exp(-(v+40)/10)));
betaM = 4*exp(-(v+65)/18);

alphaH = 0.07*exp(-(v+65)/20);
betaH = 1./(exp(-(v+35)/10)+1);

iNa = gNa*(m^3)*h*(v-vNa);

% K+ Equations and Currents
alphaN = 0.01*((v+55)/(1-exp(-(v+55)/10)));
betaN = 0.125*exp(-(v+65)/80);

iK = gK*(n^4)*(v-vK);

% Leak current
iL = gL*(v-vL);

if (t>=from && t<=to)
    iap=Iapp;
else
    iap=0;
end

output(1,1)=(-iNa-iK-iL+iap)/C;
output(2,1)=alphaM*(1-m)-betaM*m;
output(3,1)=alphaH*(1-h)-betaH*h;
output(4,1)=alphaN*(1-n)-betaN*n;



