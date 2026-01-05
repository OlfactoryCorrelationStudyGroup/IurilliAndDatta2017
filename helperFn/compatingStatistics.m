%% Mean fire rate change from baseline

results_pcx = cell(size(x_pcx));

for i = 1:numel(x_pcx)
    X = x_pcx{i};                      % numeric matrix
    results_pcx{i} = meanResponseMagnitude(X);  % [n_trials × 1]
end

results_plcoa = cell(size(x_plcoa));

for i = 1:numel(x_plcoa)
    X = x_plcoa{i};
    results_plcoa{i} = meanResponseMagnitude(X);
end


all_pcx_mean = vertcat(results_pcx{:});
all_plcoa_mean = vertcat(results_plcoa{:});

deltaMean = mean(all_plcoa_mean) - mean (all_pcx_mean);

%% Fraction of neurons responding

results_pcx = cell(size(x_pcx));

for i = 1:numel(x_pcx)
    X = x_pcx{i};                      % numeric matrix
    results_pcx{i} = fractionResponsiveNeurons(X,1);  % [n_trials × 1]
end

results_plcoa = cell(size(x_plcoa));

for i = 1:numel(x_plcoa)
    X = x_plcoa{i};
    results_plcoa{i} = fractionResponsiveNeurons(X,1);
end


all_pcx_fraction = vertcat(results_pcx{:});
all_plcoa_fraction = vertcat(results_plcoa{:});

deltaFraction = mean(all_plcoa_fraction) - mean (all_pcx_fraction);

%% Comparing Variance

results_pcx = cell(size(x_pcx));

for i = 1:numel(x_pcx)
    X = x_pcx{i};                      % numeric matrix
    results_pcx{i} = responseVariance(X);  % [n_trials × 1]
end

results_plcoa = cell(size(x_plcoa));

for i = 1:numel(x_plcoa)
    X = x_plcoa{i};
    results_plcoa{i} = responseVariance(X);
end


all_pcx_Var = vertcat(results_pcx{:});
all_plcoa_Var = vertcat(results_plcoa{:});

deltaVar = mean(all_plcoa_Var) - mean (all_pcx_Var);

fprintf('Mean response magnitude plCoA: %.4f\n', mean(all_plcoa_mean));
fprintf('Mean response magnitude Pcx: %.4f\n', mean(all_pcx_mean));
fprintf('Mean response fraction plCoA: %.4f\n', mean(all_plcoa_fraction));
fprintf('Mean response fraction Pcx: %.4f\n', mean(all_pcx_fraction));
fprintf('Mean response variance plCoA: %.4f\n', mean(all_plcoa_Var));
fprintf('Mean response variance Pcx: %.4f\n', mean(all_pcx_Var));

%% single variable classifier using Variance

% Combine data
X = [all_pcx_Var(:); all_plcoa_Var(:)];   % feature: variance
y = [zeros(numel(all_pcx_Var),1); ...
     ones(numel(all_plcoa_Var),1)];       % labels: 0 = PCx, 1 = plCoA

% Compute decision threshold (midpoint of means)
mu_pcx   = mean(all_pcx_Var);
mu_plcoa = mean(all_plcoa_Var);
theta = (mu_pcx + mu_plcoa) / 2;

% Predict
y_hat = X > theta;

% Accuracy
accuracy_simple = mean(y_hat == y);

fprintf('Simple variance threshold accuracy: %.2f%%\n', ...
        accuracy_simple * 100);


%% Cross validation classifier

cv = cvpartition(y, 'KFold', 5);
acc = zeros(cv.NumTestSets,1);

for fold = 1:cv.NumTestSets
    trainIdx = training(cv, fold);
    testIdx  = test(cv, fold);

    % Compute threshold on training data only
    mu0 = mean(X(trainIdx & y==0));   % PCx
    mu1 = mean(X(trainIdx & y==1));   % plCoA
    theta = (mu0 + mu1) / 2;

    % Predict test set
    y_pred = X(testIdx) > theta;

    acc(fold) = mean(y_pred == y(testIdx));
end

fprintf('Cross-validated accuracy (variance only): %.2f ± %.2f %%\n', ...
        mean(acc)*100, std(acc)*100);
