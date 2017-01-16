function net = dbm_learn(data,net,opts,epoch)
    
    % Contrastive divergence learning.
    %
    % USAGE: net = dbm_learn(data,net,opts,[epoch])
    %
    % INPUTS:
    %   data - [nCases x nFeatures x nBatches] observed vectors
    %   net - network structure (see dbm_init)
    %   opts - options structure
    %   epoch (optional) - starting epoch (default: 1)
    %
    % OUTPUTS:
    %   net - trained network
    %
    % Sam Gershman, June 2013
    
    nCases = size(data,1);
    nBatches = size(data,3);
    
    if nargin < 4; epoch = 1; end
    t = (epoch-1)*nBatches;
    nLayers = length(net.nUnits);
    clamped_pos = net.layer==1;
    clamped_neg = false(size(clamped_pos));
    
    % initialize unit activations
    mu_pos = zeros(nCases,sum(net.nUnits));
    for i = 1:nLayers
        mu_pos(:,net.layer==i) = dbm_act(randn(nCases,net.nUnits(i)),net.unit_type{i},false);
    end
    
    for epoch = epoch:opts.nEpochs
        
        disp(['epoch ',num2str(epoch)]);
        
        for batch = randperm(nBatches)
            
            t = t + 1;
            lrate = opts.lrate(t);  % learning rate
            lrate = max(lrate,0.0001);
            
            %-------- positive phase --------------%
            mu_pos(:,clamped_pos) = squeeze(data(:,:,batch));
            mu_pos = dbm_infer(mu_pos,clamped_pos,false,net,opts.nMF); % mean-field updates
            
            %-------- negative phase --------------%
            mu_neg = dbm_infer(mu_pos,clamped_neg,true,net,opts.nGibbs); % Gibbs updates
            
            %-------- update weights and biases ---%
            net.b = net.b + lrate*(mean(mu_pos)-mean(mu_neg));
            for i = 1:nLayers-1
                ix1 = net.layer==i;
                ix2 = net.layer==(i+1);
                CD = mu_pos(:,ix1)'*mu_pos(:,ix2) - mu_neg(:,ix1)'*mu_neg(:,ix2);
                net.W(ix1,ix2) = net.W(ix1,ix2) + lrate*(CD/nCases - opts.weightcost*net.W(ix1,ix2));
                net.W(ix2,ix1) = net.W(ix1,ix2)';
            end
        end
        if ~isempty(opts.savefile); save(opts.savefile,'net'); end
    end