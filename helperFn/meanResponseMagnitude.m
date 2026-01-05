function mu = meanResponseMagnitude(X)
% X: [n_trials × n_neurons] firing-rate change matrix
% mu: [n_trials × 1] mean response per trial

    mu = mean(X, 2);   % average across neurons
end
