function [NeuronCount] = CountNeurons_Iurilli(espe)
%This functions counts how many neuron the data has - fits the format of
%Iurilli & Datton 2017
%   Input is espe data struct and output in neuron count
NeuronCount = int32(0);
for i = 1:size(espe,2)

    for j= 1:size(espe(i).shank,2)
        if isempty(espe(i).shank(j).SUA) % Shank didnt recond any single units
            continue
        end

            NeuronCount = NeuronCount + int32(size(espe(i).shank(j).SUA.cell,2));
        
    end
end
end

%size(espe(1).shank(1).SUA.cell,2)