function net = dbm_init(nUnits,unit_type)
    
    % Initialize network.
    %
    % USAGE: net = dbm_init(nUnits,unit_type)
    %
    % INPUTS:
    %   nUnits - [1 x nLayers] number of units in each layer (first element
    %             is the number of features in the observations)
    %   unit_type - [1 x nLayers] cell array of strings indicating what
    %             type of unit ('bernoulli' or 'gaussian') in each layer
    %
    % OUTPUTS:
    %   net - initialized network, with the following fields:
    %           .layer - [1 x nUnits] which layer each unit belongs to
    %           .W - [nUnits x nUnits] connection weights
    %           .b - [1 x nUnits] biases
    %           .nUnits - [1 x nLayers] number of units in each layer
    %           .unit_type - [1 x nLayers] type of unit in each layer
    %
    % Sam Gershman, June 2013
    
    nLayers = length(nUnits);       % number of layers
    N = sum(nUnits);                % total number of units
    
    net.layer = [];                 % layer indices
    net.W = sparse(zeros(N));       % weight matrix
    net.b = zeros(1,N);             % biases
    net.nUnits = nUnits;
    net.unit_type = unit_type;
    
    % compute layer indices
    for i = 1:nLayers
        net.layer = [net.layer zeros(1,nUnits(i))+i];
    end
    
    % initialize weight matrix
    for i = 1:nLayers-1
        ix1 = net.layer==i;
        ix2 = net.layer==(i+1);
        net.W(ix1,ix2) = randn(nUnits(i),nUnits(i+1))./max(nUnits(i:i+1));
        net.W(ix2,ix1) = net.W(ix1,ix2)';
    end