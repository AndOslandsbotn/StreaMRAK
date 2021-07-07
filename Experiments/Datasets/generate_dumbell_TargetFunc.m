clear all; close all; clc;

%% generate a target function on top of the dataset
currfolder = pwd;
filepath = strcat(pwd, '\Dumbell');
filename = strcat('twoblobTr', '.csv');
Xtr = readmatrix(fullfile(filepath, filename));
filename = strcat('twoblobTs', '.csv');
Xts = readmatrix(fullfile(filepath, filename));


[ftr] = constantInHighDimSinusInLowDim(Xtr);
[fts] = constantInHighDimSinusInLowDim(Xts);
filename = strcat('complexSinusTr' ,'.csv');
csvwrite(fullfile(filepath, filename),ftr);
filename = strcat('complexSinusTs', '.csv');
csvwrite(fullfile(filepath, filename),fts);

plot(Xtr, ftr)

[~,ind] = sort(Xtr(:,1));
XtrSort = Xtr(ind,:);
ftr = ftr(ind,:);
plot(XtrSort(:,1),ftr)

function [f] = constantInHighDimSinusInLowDim(X)
% This function is adapted for the twoblob dataset

indOutLower = X(:,1) <= 1;
indOutUpper = X(:,1) >= 3;

C1 = 1;
a = -1; %

b = C1-a;
C3 = b + 3*a;

% Choosing parameters such that f is continously differentiable to first
% derivative
n1 = 2; % frequency %4
phi1 = - pi*n1;

n2 = 7% frequency %4
phi2 = - pi*n2;

B1 = -phi1;
A1 = -a/B1;

B2 = -phi2;
A2 = -a/B2;

eps = 10^(-4)*rand(size(X(:,1)));
f = A1*sin(B1*X(:,1)+ phi1).*cos(B2*X(:,1)+ phi2) + (a*X(:,1) + b);

f(indOutLower) = C1; 
f(indOutUpper) = C3;
end

