%% DMAIC CASE STUDY 2: NASA C-MAPSS (FD001) — Predictive Maintenance & Resilience
%
% Author: Akshay Thummalapally
% Date: August 2025
%
% Purpose:
% Implements a clean, reproducible DMAIC pipeline on the C-MAPSS FD001 dataset.
% - Loads config from /code/configcode.m
% - Loads train/test/RUL data
% - DEFINE & MEASURE: compute RUL, baseline MTBF & distributions
% - ANALYZE: feature screening (constant sensors), correlations, baseline linear model
% - IMPROVE: simple PdM trigger; illustrative Weibull-based MTBF uplift
% - CONTROL: SPC chart of residuals; control plan outline
% - Saves ALL figures/tables into /output/
%
% How to run (from repo root or anywhere in MATLAB):
%   >> run(fullfile('code','cmapss_dmaic.m'))
%
% Requirements:
% - MATLAB R2021a+ (recommended)
% - Statistics and Machine Learning Toolbox
% - (Optional) Predictive Maintenance Toolbox (for extras; not required here)

clear; clc; close all;

%% 0) CONFIG & PATHS
cfg = configcode();  % user-editable configuration in /code/configcode.m

% Resolve folders relative to /code/
thisFileDir = fileparts(mfilename('fullpath'));     % /code
repoRoot    = fileparts(thisFileDir);               % repo root
dataDir     = fullfile(thisFileDir, cfg.dataset_dir); % typically '../data' relative to /code
outDir      = fullfile(repoRoot, 'output');         % always write here

if ~exist(outDir,'dir')
    mkdir(outDir);
end

fprintf('[INFO] Output directory: %s\n', outDir);
fprintf('[INFO] Data directory:   %s\n', dataDir);

%% 1) LOAD DATA (DEFINE & MEASURE)
fprintf('[PHASE 1] DEFINE & MEASURE\n');

% C-MAPSS FD001 has 26 columns (1..26)
colNames = {'UnitNumber','TimeCycle','OpSet1','OpSet2','OpSet3', ...
    'Sensor1','Sensor2','Sensor3','Sensor4','Sensor5','Sensor6','Sensor7', ...
    'Sensor8','Sensor9','Sensor10','Sensor11','Sensor12','Sensor13', ...
    'Sensor14','Sensor15','Sensor16','Sensor17','Sensor18','Sensor19', ...
    'Sensor20','Sensor21'};

% --- Safe loaders with clear errors ---
trainPath = fullfile(dataDir, cfg.train_file);
testPath  = fullfile(dataDir, cfg.test_file);
rulPath   = fullfile(dataDir, cfg.rul_file);

try
    trainData = readtable(trainPath, 'FileType','text', 'Delimiter',' ');
    trainData.Properties.VariableNames = colNames;

    testData  = readtable(testPath,  'FileType','text', 'Delimiter',' ');
    testData.Properties.VariableNames = colNames;

    trueRUL   = readtable(rulPath,   'FileType','text');
    trueRUL.Properties.VariableNames = {'TrueRUL'};

    fprintf('[OK] Datasets loaded: %s, %s, %s\n', cfg.train_file, cfg.test_file, cfg.rul_file);
catch ME
    error(['[ERROR] Data load failed: %s\n' ...
        'Ensure files exist under dataDir and the schema matches FD001.'], ME.message);
end

% Compute RUL for training (runs-to-failure)
maxCycles = grpstats(trainData.TimeCycle, trainData.UnitNumber, 'max');
maxCyclesTable = table(unique(trainData.UnitNumber), maxCycles, ...
    'VariableNames', {'UnitNumber','MaxCycle'});

trainData = outerjoin(trainData, maxCyclesTable, 'Keys','UnitNumber','MergeKeys',true);
trainData.RUL = trainData.MaxCycle - trainData.TimeCycle;
trainData.MaxCycle = [];

% --- Visuals: Representative sensor degradation (Sensors 7 & 12 for units 1..5) ---
fA = figure('Name','Figure A: Representative Sensor Degradation','NumberTitle','off');
subplot(2,1,1); hold on;
for i = 1:5
    ed = trainData(trainData.UnitNumber==i,:);
    plot(ed.TimeCycle, ed.Sensor7);
end
hold off; grid on;
title('Figure A: Sensor 7 Trajectories');
xlabel('Time (Cycles)'); ylabel('Sensor 7');

subplot(2,1,2); hold on;
for i = 1:5
    ed = trainData(trainData.UnitNumber==i,:);
    plot(ed.TimeCycle, ed.Sensor12);
end
hold off; grid on;
title('Sensor 12 Trajectories');
xlabel('Time (Cycles)'); ylabel('Sensor 12');
saveas(fA, fullfile(outDir,'Figure_A_Sensor_Degradation.png'));
close(fA);

% Baseline failure-time distribution (training set)
failureTimes = maxCyclesTable.MaxCycle;
fB = figure('Name','Figure B: Histogram of Engine Failure Times','NumberTitle','off');
histogram(failureTimes, 15); grid on;
title('Figure B: Baseline Distribution of Engine Failure Times');
xlabel('Time to Failure (Cycles)'); ylabel('Frequency');
saveas(fB, fullfile(outDir,'Figure_B_Failure_Histogram.png'));
close(fB);

baselineMTBF = mean(failureTimes);
fprintf('[INFO] Baseline MTBF: %.2f cycles\n', baselineMTBF);

%% 2) ANALYZE — PREDICTIVE SIGNALS & BASELINE MODEL
fprintf('[PHASE 2] ANALYZE\n');

% Remove near-constant sensors (threshold in cfg)
sensorVars = trainData.Properties.VariableNames(6:26); % Sensor1..Sensor21
sensorStd  = std(trainData{:,sensorVars});
constantSensors = sensorVars(sensorStd < cfg.constant_threshold);

if ~isempty(constantSensors)
    trainData(:, constantSensors) = [];
    testData(:,  constantSensors) = [];
    fprintf('[INFO] Dropped near-constant sensors: %s\n', strjoin(constantSensors, ', '));
else
    fprintf('[INFO] No near-constant sensors detected under threshold %.1e\n', cfg.constant_threshold);
end

% Refresh predictor list (OpSets + remaining sensors; exclude: UnitNumber, TimeCycle, RUL)
allVars = trainData.Properties.VariableNames;
predictorMask = ~ismember(allVars, {'UnitNumber','TimeCycle','RUL'});
predictorVars = allVars(predictorMask);

% Rank features by absolute Spearman correlation to RUL
corrWithRUL = nan(numel(predictorVars),1);
for i = 1:numel(predictorVars)
    corrWithRUL(i) = corr(trainData.(predictorVars{i}), trainData.RUL, ...
                          'type','Spearman', 'rows','complete');
end
corrTable = table(predictorVars', abs(corrWithRUL), ...
    'VariableNames', {'Feature','AbsSpearmanRUL'});
corrTable = sortrows(corrTable, 'AbsSpearmanRUL', 'descend');

% Save correlation table
writetable(corrTable, fullfile(outDir,'Table_A_Sensor_Correlation.csv'));
fprintf('[OK] Saved Table_A_Sensor_Correlation.csv (top features by |Spearman|)\n');

% Choose top-K predictors (from config)
kTop = min(cfg.top_k_features, height(corrTable));
topPredictors = corrTable.Feature(1:kTop)';

% Fit baseline linear regression: RUL ~ top predictors
formula = ['RUL ~ ' strjoin(topPredictors,' + ')];
rulModel = fitlm(trainData, formula);

% --- Build last-cycle rows per test engine ---
testSorted = sortrows(testData, {'UnitNumber','TimeCycle'});
isLastRowOfGroup = [diff(testSorted.UnitNumber)~=0; true];
lastCycleTestData = testSorted(isLastRowOfGroup,:);
lastCycleTestData = sortrows(lastCycleTestData, 'UnitNumber');

assert(height(lastCycleTestData) == height(trueRUL), ...
    'Mismatch: number of test engines and RUL rows differ.');
assert(issorted(lastCycleTestData.UnitNumber), ...
    'UnitNumber is not sorted ascending in lastCycleTestData.');

% Predict RUL at last observed point
X_test = lastCycleTestData(:, topPredictors);
predictedRUL = predict(rulModel, X_test);

% Performance metrics
RMSE = sqrt(mean((predictedRUL - trueRUL.TrueRUL).^2));
MAE  = mean(abs(predictedRUL - trueRUL.TrueRUL));
fprintf('[INFO] Test Performance — RMSE = %.2f, MAE = %.2f\n', RMSE, MAE);

% Predicted vs Actual scatter
fC = figure('Name','Figure C: Predicted vs. Actual RUL','NumberTitle','off');
scatter(trueRUL.TrueRUL, predictedRUL, 'filled'); hold on;
mx = max([trueRUL.TrueRUL; predictedRUL]);
plot([0 mx],[0 mx],'r--','LineWidth',2);
grid on; hold off;
title(sprintf('Figure C: Predicted vs. Actual RUL (RMSE = %.2f)', RMSE));
xlabel('True RUL (Cycles)'); ylabel('Predicted RUL (Cycles)');
legend('Predictions','Perfect Prediction','Location','northwest');
saveas(fC, fullfile(outDir,'Figure_C_Predicted_vs_Actual.png'));
close(fC);

%% 3) IMPROVE — SIMPLE PdM TRIGGER & RESILIENCE UPLIFT
fprintf('[PHASE 3] IMPROVE\n');

% Policy: perform maintenance if predicted RUL < threshold (from cfg)
threshold = cfg.maintenance_threshold;
fprintf('[INFO] PdM Policy: Replace component if predicted RUL < %d cycles.\n', threshold);

unplanned_before = height(trueRUL);
missed = sum(trueRUL.TrueRUL < threshold & predictedRUL >= threshold); % missed detections
unplanned_after = missed;

downtimeReduction = (unplanned_before - unplanned_after) / unplanned_before * 100;
fprintf('[INFO] Unplanned Failures — Before: %d | After: %d | Reduction: %.2f%%\n', ...
    unplanned_before, unplanned_after, downtimeReduction);

% Weibull fit to baseline failure times (training) and illustrative uplift
pd = fitdist(failureTimes, 'Weibull');
beta_before = pd.B;  % shape
eta_before  = pd.A;  % scale
eta_after   = eta_before * cfg.eta_uplift_factor; % e.g., +15% characteristic life

mtbf_before = eta_before * gamma(1 + 1/beta_before);
mtbf_after  = eta_after  * gamma(1 + 1/beta_before);

resilienceResults = table( ...
    ["Baseline"; "After PdM"]', ...
    [mtbf_before; mtbf_after]', ...
    'VariableNames', {'State','MTBF_Cycles'});

writetable(resilienceResults, fullfile(outDir,'Table_B_Resilience_Improvement.csv'));
fprintf('[OK] Saved Table_B_Resilience_Improvement.csv\n');

%% 4) CONTROL — SPC & CONTROL PLAN
fprintf('[PHASE 4] CONTROL\n');

% SPC of residuals
residuals = trueRUL.TrueRUL - predictedRUL;
fD = figure('Name','Figure D: SPC Chart of Model Residuals','NumberTitle','off');
controlchart(residuals, 'charttype','i');
title('Figure D: Control Chart of RUL Prediction Residuals');
xlabel('Engine Unit (Test Set)'); ylabel('Prediction Error (Cycles)');
saveas(fD, fullfile(outDir,'Figure_D_SPC_Residuals.png'));
close(fD);

% Control plan (text file, consumable in repo)
planLines = [
    "- Monitor RUL predictions each cycle; alert if Predicted RUL < " + string(threshold) + " cycles."
    "- Review SPC chart of residuals quarterly; retrain if out-of-control."
    "- Provide dashboard: Predicted RUL vs Time per engine for operator support."
];
fid = fopen(fullfile(outDir,'Control_Plan.txt'),'w');
cleanupObj = onCleanup(@() fclose(fid));
fprintf(fid, "CONTROL PLAN\n------------\n");
fprintf(fid, "%s\n", planLines);
clear cleanupObj;

%% 5) SAVE MODEL SUMMARY & RUN METADATA
% Store key run info for traceability
modelSummaryPath = fullfile(outDir,'Model_Summary.txt');
fid2 = fopen(modelSummaryPath,'w');
cleanupObj2 = onCleanup(@() fclose(fid2));
fprintf(fid2, "Baseline Linear Model (RUL ~ top-%d features)\n", kTop);
fprintf(fid2, "Features: %s\n\n", strjoin(topPredictors, ', '));
fprintf(fid2, "Test RMSE: %.4f\nTest MAE: %.4f\n\n", RMSE, MAE);
fprintf(fid2, "Weibull (baseline): beta=%.4f, eta=%.4f\n", beta_before, eta_before);
fprintf(fid2, "MTBF (baseline): %.4f | MTBF (after PdM): %.4f\n", mtbf_before, mtbf_after);
fprintf(fid2, "Downtime reduction (%%): %.2f\n", downtimeReduction);
clear cleanupObj2;

fprintf('\n[DONE] DMAIC analysis complete. All outputs saved under: %s\n', outDir);
