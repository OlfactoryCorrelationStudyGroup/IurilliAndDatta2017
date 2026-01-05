function zResp = firingRateChange(spikeMat, baselineWin, odorWin, dt)
% spikeMat: trials × time (logical or numeric)

% Baseline firing rate per trial
baselineRates = sum(spikeMat(:, baselineWin), 2) / (numel(baselineWin) * dt);

% Odor firing rate per trial
odorRates = sum(spikeMat(:, odorWin), 2) / (numel(odorWin) * dt);

% Baseline statistics (per neuron)
mu = mean(baselineRates);
sigma = std(baselineRates);

% Numerical safety
if sigma == 0
    zResp = zeros(size(odorRates));
else
    zResp = (odorRates - mu) ./ sigma;
end
end
