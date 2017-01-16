function batchdata = dbm_make_batches(data,nBatches)
    
    % Convert data to batches.
    %
    % USAGE: batchdata = dbm_make_batches(data,nBatches)
    %
    % INPUTS:
    %   data - [nObservations x nFeatures] observed vectors
    %   nBatches - number of batches
    %
    % OUTPUTS:
    %   batchdata - [nCases x nFeatures x nBatches] observed vectors
    %
    % Sam Gershman, June 2013
    
    [N D] = size(data);
    batchsize = floor(N/nBatches);
    batchdata = zeros(batchsize,D,nBatches);
    
    k = 1;
    for b = 1:nBatches
        batchdata(:,:,b) = data(k:(k+batchsize-1),:);
        k = k + batchsize;
    end