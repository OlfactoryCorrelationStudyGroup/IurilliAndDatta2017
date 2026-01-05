function [X_region] = buildRegionMatrix(espe, baselineWin, odorWin, dt)

X_region = {};

for i = 1:numel(espe)   % loop over experiments / sessions
    
    trialFeatures = [];   % will be trials × neurons for this session
    
    for j = 1:numel(espe(i).shank)
        
        SUA = espe(i).shank(j).SUA;
        if isempty(SUA) || ~isfield(SUA,'cell')
            continue
        end
        
        for k = 1:numel(SUA.cell)
            neuron_responses = [];

            for o = 1:numel(SUA.cell(k).odor)

            % Access spike matrix
            spikeMat = SUA.cell(k).odor(o).spikeMatrix;  
            
            % Compute firing-rate change
            deltaRate = firingRateChange(spikeMat, baselineWin, odorWin, dt);
            deltaRate = full(deltaRate);
            neuron_responses = [neuron_responses; deltaRate];
            end

            % Append as a feature (column)
            trialFeatures = [trialFeatures, neuron_responses];

        end
    end
    
    % Append session trials
    X_region{end + 1} = trialFeatures;
end

end
