% This code uses buildRegionCellArrays and fireRateChange to get to cell
% arrayes of neuronal resposnse. each cell is a trial X neurons matrix of
% fire rate change in rßesponse to odor exposure in z scores. Odor identity
% is implicit.
% Defining veribles
espe_plcoa = load ("/Users/ofekh/Library/CloudStorage/OneDrive-Bar-IlanUniversity-Students/MachineLearning_HW/Final Project data/plCoA_15.mat");
espe_pcx = load('/Users/ofekh/Library/CloudStorage/OneDrive-Bar-IlanUniversity-Students/MachineLearning_HW/Final Project data/aPCx_15.mat');
espe_plcoa = espe_plcoa.espe;
espe_pcx = espe_pcx.espe;
baselineWin = 2000:4000;
odorWin = 4000:6000;
dt = 1e-3;

[x_pcx , x_plcoa] = buildRegionCellArrays(espe_pcx,espe_plcoa,baselineWin,odorWin,dt);

save('region_sessions.mat',"x_pcx","x_plcoa",'-v7.3');

