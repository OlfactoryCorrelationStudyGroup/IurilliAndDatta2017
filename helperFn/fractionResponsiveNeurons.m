function frac = fractionResponsiveNeurons(X, threshold)
% X: [n_trials × n_neurons]
% threshold: scalar (e.g., 1 for z-score)
% frac: [n_trials × 1]

    responsive = abs(X) > threshold;
    frac = mean(responsive, 2);
end
