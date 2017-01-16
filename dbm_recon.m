function mu = dbm_recon(data,observed,net,nIter)
    
    % Reconstruct an unobserved subset of the data given an observed
    % subset.
    %
    % USAGE: mu = dbm_recon(data,observed,net,[nIter])
    %
    % INPUTS:
    %   data - [nCases x nFeatures x nBatches] observed vectors
    %   observed - indices of observed features
    %   net - trained network structure (see dbm_learn)
    %   nIter (optional) - number of Gibbs iterations (default: 50)
    %
    % OUTPUTS:
    %   mu - [nCases x nFeatures x nBatches] reconstructed data (observed
    %       features are set to their observed values; missing features are set
    %       to their predicted values)
    %
    % Sam Gershman, June 2013
    
    
    if nargin < 4; nIter = 50; end
    
    sample = false;
    nCases = size(data,1);
    
    s = zeros(nCases,sum(net.nUnits));
    for i = 1:length(net.nUnits)
        s(:,net.layer==i) = dbm_act(randn(nCases,net.nUnits(i)),net.unit_type{i},sample);
    end
    
    ix = find(net.layer==1); ix = ix(observed);
    clamped = false(1,size(s,2)); clamped(ix) = 1;
    mu = data;
    for b = 1:size(data,3)
        disp(['batch ',num2str(b)]);
        s(:,ix) = data(:,observed,b);
        x = dbm_infer(s,clamped,sample,net,nIter);
        mu(:,:,b) = x(:,net.layer==1);
    end