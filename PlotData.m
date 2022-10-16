close all; clearvars; clc;

%upperDir is path to "upper directory" of where our data is:
upperDir = 'C:\Users\lujain\Desktop\Bowie\Datasim1\'; % CHANGE this to your path
% this is the path of where your code/functions exist:
addpath('C:\Users\lujain\Desktop\Bowie\'); % CHANGE this to your path

% Frame to plot
frame = 20; %In the given data folder there are 22 frames, I chose 18 here

% Load density data
ne_bg = load([upperDir '/ne_unpert.txt']);  %this is my background density
ne = load([upperDir '/ne' num2str(frame) '.txt']); %this is my total denisty

% Load potential data
phi_bg = load([upperDir 'phi_unpert.txt']); %background
phi = load([upperDir '/phi' num2str(frame) '.txt']); %total

% Load X and Y data
XX = load([upperDir 'X.txt']);
YY = load([upperDir 'Y.txt']);

% Plotting:
figure;
pcolor(XX,YY,ne);
shading flat;
colormap inferno;
colorbar;


