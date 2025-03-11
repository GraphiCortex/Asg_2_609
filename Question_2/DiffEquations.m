function [output] = DiffEquations(time,init)

global eNa eK eL ...
    gNa gK gL gCaL ...
    tauNbar tauH Ca_ex RTF ...
    thetaM sigmaM thetaN sigmaN thetaS sigmaS ...
    f iapp eps kca C ...
    from to

n=init(1);
h=init(2);
ca=init(3);
v=init(4);

% Na+ and K+ Equations and Currents
minf = 1/(1+exp((v-thetaM)/sigmaM));
ninf = 1/(1+exp((v-thetaN)/sigmaN));
tauN = tauNbar./cosh((v-thetaN)/(2*sigmaN));

alphaH = 0.128*exp(-(v+50)/18);
betaH = 4/(1+exp(-(v+27)/5));
hinf = alphaH/(alphaH+betaH);

iNa = gNa*(minf^3)*h*(v-eNa);
iK = gK*(n^4)*(v-eK);

% L-Type Ca++ Equations and Current
sinf = 1/(1+exp((v-thetaS)/sigmaS));
iCaL = gCaL*(sinf^2)*v*(Ca_ex/(1-exp((2*v)/RTF)));


% Leak current
iL = gL*(v-eL);

if(time>=from&&time<=to)
    iap=iapp;
else
    iap=0;
end

output(1,1)=(ninf-n)/tauN;
output(2,1)=(hinf-h)/tauH;
output(3,1)=-f*(eps*(iCaL)+ kca*(ca-0.1));
output(4,1)=(-iNa-iK-iCaL-iL+iap)/C;

