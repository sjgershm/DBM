function s = dbm_infer(s,clamped,sample,net,nIter,layers)
    
    % Gibbs or mean-field inference conditional on a set of clamped units.
    %
    % USAGE: s = dbm_infer(s,clamped,sample,net,[nIter],[layers])
    %
    % INPUTS:
    %   s - [nCases x nUnits] initial activations
    %   clamped - [1 x nUnits] logical index vector specifying which units are clamped
    %   sample - {0,1} use Gibbs sampling (1) or mean-field (0)
    %   nIter (optional) - number of iterations (default: 1)
    %   layers (optional) - which layers to infer (default: all)
    %
    % OUTPUTS:
    %   s - final activations
    %
    % Sam Gershman, June 2013
    
    if nargin < 5; nIter = 1; end
    if nargin < 6; layers = 1:length(net.nUnits); end
    
    for i = 1:nIter
        for layer = layers
            ix = net.layer==layer & ~clamped;                   % which units to update
            if any(ix)
                input = bsxfun(@plus,s(:,~ix)*net.W(~ix,ix),net.b(ix));
                s(:,ix) = dbm_act(input,net.unit_type{layer},sample);
            end
        end
    end