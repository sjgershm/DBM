function net = dbm_pretrain(data,net,opts)
    
    % Layerwise pretraining of a deep Boltzmann machine.
    %
    % USAGE: net = dbm_pretrain(data,net,opts)
    %
    % INPUTS:
    %   data - [nCases x nFeatures x nBatches] observed vectors
    %   net - network structure (see dbm_init)
    %   opts - options structure
    %
    % OUTPUTS:
    %   net - updated network structure
    %
    % Sam Gershman, June 2013
    
    nCases = size(data,1);
    nLayers = length(net.nUnits);
    clamped = net.layer==1;
    
    % initialize unit activations
    mu_pos = zeros(nCases,sum(net.nUnits));
    for i = 1:nLayers
        mu_pos(:,net.layer==i) = dbm_act(randn(nCases,net.nUnits(i)),net.unit_type{i},false);
    end
    
    for layer = 1:nLayers-1
        
        % weight multipliers
        if layer == 1
            m1 = 2; m2 = 1;
        elseif layer == nLayers-1
            m1 = 1; m2 = 2;
        else
            m1 = 2; m2 = 2;
        end
        
        t = 0;
        ix1 = net.layer==layer;
        ix2 = net.layer==(layer+1);
        
        for epoch = 1:opts.nEpochs_pretrain
            
            disp(['layer ',num2str(layer),', pretrain epoch ',num2str(epoch)]);
            
            for batch = randperm(size(data,3))
                
                t = t + 1;
                lrate = opts.lrate_pretrain(t);  % learning rate
                
                %-------- positive phase --------------%
                mu_pos(:,clamped) = squeeze(data(:,:,batch));
                mu_pos = dbm_infer(mu_pos,clamped,true,net,1,1:layer);
                input = bsxfun(@plus,m1*mu_pos(:,ix2)*net.W(ix2,ix1),net.b(ix1));
                mu_pos(:,ix1) = dbm_act(input,net.unit_type{layer},false);
                
                %-------- negative phase --------------%
                mu_neg = dbm_infer(mu_pos,clamped,true,net,1,1:layer);
                input = bsxfun(@plus,m2*mu_neg(:,ix1)*net.W(ix1,ix2),net.b(ix2));
                mu_neg(:,ix2) = dbm_act(input,net.unit_type{layer+1},true);
                input = bsxfun(@plus,m1*mu_neg(:,ix2)*net.W(ix2,ix1),net.b(ix1));
                mu_neg(:,ix1) = dbm_act(input,net.unit_type{layer},false);
                
                
                %-------- update weights and biases ---%
                net.b(ix1) = net.b(ix1) + lrate*(mean(mu_pos(:,ix1))-mean(mu_neg(:,ix1)));
                CD = mu_pos(:,ix1)'*mu_pos(:,ix2)- mu_neg(:,ix1)'*mu_neg(:,ix2);
                net.W(ix1,ix2) = net.W(ix1,ix2) + lrate*(CD/nCases - opts.weightcost_pretrain*net.W(ix1,ix2));
                net.W(ix2,ix1) = net.W(ix1,ix2)';
            end
        end
        if ~isempty(opts.savefile); save([opts.savefile,'_pretrain'],'net'); end
    end