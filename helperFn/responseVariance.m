function v = responseVariance(X)
% X: [n_trials × n_neurons]
% v: [n_trials × 1] variance across neurons

    v = var(X, 0, 2);  % variance across columns
end
