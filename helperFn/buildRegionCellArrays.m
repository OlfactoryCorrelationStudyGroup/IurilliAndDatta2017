function [X_pcx, X_plcoa] = buildRegionCellArrays( ...
                    espe_pcx, espe_plcoa, ...
                    baselineWin, odorWin, dt)
X_plcoa = buildRegionMatrix(espe_plcoa,baselineWin,odorWin,dt);
X_pcx = buildRegionMatrix(espe_pcx,baselineWin,odorWin,dt);

end