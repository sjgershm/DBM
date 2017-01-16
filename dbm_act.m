function output = dbm_act(input,unit_type,sample)
    
    % Compute activations given inputs.
    %
    % USAGE: output = dbm_act(input,unit_type,[sample])
    %
    % Sam Gershman, June 2013
    
    if nargin < 3; sample = false; end
    
    switch unit_type
        case 'gaussian'
            if sample
                output = normrnd(input,0.1);
            else
                output = input;
            end
        case 'bernoulli'
            input = 1./(1+exp(-input));
            if sample
                output = double(rand(size(input)) < input);
            else
                output = input;
            end
    end