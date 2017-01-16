% Demo code for deep Boltzmann machine
%
% Sam Gershman, June 2013

clear

% Load Olivetti faces and convert to batches
load olivettifaces
data = dbm_make_batches(zscore(faces)',4);
nFeatures = size(data,2);

% Set options
opts.nEpochs = 50;  % in general, this should be much larger (e.g., 300-500)
opts.nEpochs_pretrain = 10; % in general, this should be much larger (e.g., 50-100)
opts = dbm_opts(opts);

% Initialize network structure
nUnits = [nFeatures 100 100];
unit_type = {'gaussian' 'bernoulli' 'bernoulli'};
net = dbm_init(nUnits,unit_type);

% Layerwise pretraining
disp('pretraining...')
net = dbm_pretrain(data,net,opts);

% Contrastive divergence learning
disp('learning...')
net = dbm_learn(data,net,opts);