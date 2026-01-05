function [X, y] = buildClassifierData(X_pcx, X_plcoa, N)
% Builds ML-ready dataset by subsampling N neurons per session
%
% This function will be rebuild in python for easier resampling for SVM
% Inputs:
%   X_pcx   - cell array of PCx session matrices
%   X_plcoa - cell array of plCoA session matrices
%   N       - number of neurons to subsample per session
%
% Outputs:
%   X       - [nSamples × N] feature matrix
%   y       - [nSamples × 1] region labels (0 = PCx, 1 = plCoA)

X = [];
y = [];

%% -------PCx session-------

for session = 1:numel (X_pcx)
    Xs = X_pcx{session};

    if size (Xs,2) < N
        continue % Skips session if it has less then N neurons

    end
    
    % Randomly sample N neurons

    neuronsIdx = randperm(size(Xs,2),N);
    X_subset = Xs(:,neuronsIdx); % trials x N

    % Append to X and y

    X = [X; X_subset]; % Append the subset to the feature matrix
    y = [y; zeros(size(X_subset, 1), 1)]; % Append labels for PCx (0)

end
%% -------- plCoA sessions ----------

for session = 1:numel(X_plcoa)
    Xs = X_plcoa{session};

    if size(Xs, 2) < N
        continue % Skips session if it has less than N neurons
    end
    
    % Randomly sample N neurons
    neuronsIdx = randperm(size(Xs, 2), N);
    X_subset = Xs(:, neuronsIdx); % trials x N

    % Append to X and y
    X = [X; X_subset]; % Append the subset to the feature matrix
    y = [y; ones(size(X_subset, 1), 1)]; % Append labels for plCoA (1)
end