function opts = dbm_opts(opts)
    
    % Set default options.
    %
    % USAGE: opts = dbm_opts([opts])
    %
    % INPUTS:
    %   opts - options structure; missing or empty fields are set to defaults
    %
    % OUTPUTS:
    %   opts - updated options structure
    %
    % Sam Gershman, June 2013
    
    def_opts.lrate = @(t) 10/(2000+t);          % learning rate (training)
    def_opts.lrate_pretrain = @(t) 0.05;        % learning rate (pretraining)
    def_opts.weightcost = 0.001;                % weight decay (training)
    def_opts.weightcost_pretrain = 0.001;       % weight decay (pretraining)
    def_opts.nEpochs = 300;                     % number of epochs (training)
    def_opts.nEpochs_pretrain = 100;            % number of epochs (pretraining)
    def_opts.nMF = 10;                          % number of mean field iterations
    def_opts.nGibbs = 5;                        % number of Gibbs iterations
    def_opts.savefile = [];                     % name of file to save intermediate results (after every epoch)
    
    %set missing options
    if nargin < 1 || isempty(opts)
        opts = def_opts;
    else
        F = fieldnames(def_opts);
        for f = 1:length(F)
            if ~isfield(opts,F{f}) || (~iscell(opts.(F{f})) && isempty(opts.(F{f})))
                opts.(F{f}) = def_opts.(F{f});
            end
        end
    end