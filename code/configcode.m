function cfg = configcode()
%CONFIGCODE Centralised configuration for BOTH DMAIC case studies.
% This file is the single source of truth for paths, randomness, and
% analysis parameters across:
%   - Code 1: NASA C-MAPSS FD001 (cmapss_dmaic.m)
%   - Code 2: Steel Plates Faults (steel_plates_dmaic.m)

cfg = struct();

%% ------------------------------------------------------------------------
%  GLOBAL / REPO SETTINGS
%  (shared by both codes)
%% ------------------------------------------------------------------------
cfg.seed = 42;                 % RNG seed for deterministic runs
cfg.output_root = fullfile('..','output');  % outputs saved at repo_root/output

% If you ever want subfolders like /output/figures and /output/tables,
% set these and update scripts accordingly (currently not required):
cfg.use_output_subfolders = false;
cfg.figures_dirname = 'figures';
cfg.tables_dirname  = 'tables';


%% ------------------------------------------------------------------------
%  CODE 1: NASA C-MAPSS FD001 SETTINGS
%  Used by: cmapss_dmaic.m
%% ------------------------------------------------------------------------
% Data files (relative to cfg.cmapss_dataset_dir)
cfg.train_file = 'train_FD001.txt';
cfg.test_file  = 'test_FD001.txt';
cfg.rul_file   = 'RUL_FD001.txt';

% Where the FD001 files live (relative to /code/)
% Recommended layout: repo_root/data/...
cfg.cmapss_dataset_dir = fullfile('..','data');

% Analysis parameters
cfg.constant_threshold     = 1e-2;   % drop sensors with std < threshold
cfg.top_k_features         = 8;      % top-K by |Spearman| for baseline model
cfg.maintenance_threshold  = 30;     % PdM trigger: predicted RUL < threshold
cfg.eta_uplift_factor      = 1.15;   % illustrative Weibull eta multiplier (+15%)


%% ------------------------------------------------------------------------
%  CODE 2: STEEL PLATES FAULTS SETTINGS
%  Used by: steel_plates_dmaic.m
%% ------------------------------------------------------------------------
% Data file (relative to cfg.steel_dataset_dir)
cfg.steel_file        = 'SteelPlatesFaults_Clean.csv';

% Where the Steel Plates CSV lives (relative to /code/)
% Recommended layout: repo_root/data/...
cfg.steel_dataset_dir = fullfile('..','data');

% Target selections
cfg.steel_target_fault    = 'Other_Faults';       % binary response column
cfg.steel_continuous_ctq  = 'Sum_of_Luminosity';  % continuous CTQ for capability

% Model & evaluation
cfg.steel_shortlist_k = 12;   % number of features from univariate screen
cfg.steel_holdout     = 0.30; % test split fraction for cvpartition
cfg.steel_threshold   = 0.50; % decision threshold on predicted probability

% Control chart settings
cfg.steel_batch_size  = 50;   % p-chart subgroup size

end
