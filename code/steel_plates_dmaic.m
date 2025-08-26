%% DMAIC CASE STUDY 1: STEEL PLATES FAULTS ANALYSIS — Submission-Ready
%
% Author: Akshay Thummalapally
% Date: August 2025
%
% Purpose:
% Complete DMAIC (Define, Measure, Analyze, Improve, Control) cycle on the
% Steel Plates Faults dataset, using the shared repo layout:
%   - Config: /code/configcode.m
%   - Data:   /data/SteelPlatesFaults_Clean.csv  (configurable)
%   - Output: /output/ (auto-created)
%
% How to run (from anywhere in MATLAB):
%   >> run(fullfile('code','steel_plates_dmaic.m'))
%
% Requirements:
% - MATLAB R2021a+ recommended
% - Statistics and Machine Learning Toolbox

clear; clc; close all;

%% 0) CONFIG, PATHS, ENV, RNG
cfg = configcode();  % central config (shared with Code 1)

thisFileDir = fileparts(mfilename('fullpath'));      % /code
repoRoot    = fileparts(thisFileDir);                % repo root
dataDir     = fullfile(thisFileDir, cfg.steel_dataset_dir);  % e.g., ../data
outDir      = fullfile(repoRoot, 'output');          % shared output root

if ~exist(outDir,'dir'), mkdir(outDir); end
fprintf('[INFO] Output directory: %s\n', outDir);
fprintf('[INFO] Data directory:   %s\n', dataDir);

% Environment log (optional but useful)
v = ver; vnames = string({v.Name});
fprintf('[ENV] MATLAB: %s | Toolboxes: %s\n', version, strjoin(vnames, ', '));

% Determinism for cvpartition/binornd
rng(cfg.seed);

%% 1) SETUP & DATA LOADING (DEFINE & MEASURE)
fprintf('[PHASE 1] DEFINE & MEASURE\n');

steelPath = fullfile(dataDir, cfg.steel_file);

try
    T = readtable(steelPath);
    fprintf('[OK] Dataset loaded: %s\n', cfg.steel_file);
catch ME
    error('[ERROR] Failed to load dataset at %s\n%s', steelPath, ME.message);
end

% Basic checks
disp('First 5 rows of the dataset:');
head(T, 5);

% Feature & fault columns
featureNames = T.Properties.VariableNames(1:27);
faultNames   = T.Properties.VariableNames(28:34);

% Choose CTQ & target class
faultCounts     = sum(T{:, faultNames});
totalFaults     = sum(faultCounts);
faultPrevalence = (faultCounts / totalFaults) * 100;

prevalenceTable = table(faultNames', faultCounts', faultPrevalence', ...
    'VariableNames', {'FaultType', 'Count', 'Prevalence_Percent'});
prevalenceTable = sortrows(prevalenceTable, 'Count', 'descend');

% Save prevalence table
writetable(prevalenceTable, fullfile(outDir,'Table_0_Fault_Prevalence.csv'));

% Pareto (headless-safe)
fA = figure('Name','Figure A: Pareto of Fault Categories','NumberTitle','off','Visible','off');
pareto(faultCounts, faultNames);
title('Figure A: Pareto Chart of Fault Categories');
xlabel('Fault Type'); ylabel('Count');
saveas(fA, fullfile(outDir,'Figure_A_Pareto_Chart.png'));
close(fA);

% Selected target class & continuous CTQ
targetFault   = cfg.steel_target_fault;        % e.g., 'Other_Faults'
continuousCTQ = cfg.steel_continuous_ctq;      % e.g., 'Sum_of_Luminosity'

baselineDefectRate = (sum(T.(targetFault)) / height(T)) * 100;
fprintf('[INFO] Baseline defect rate for %s: %.2f%%\n', targetFault, baselineDefectRate);

% Proxy spec limits (10th–90th percentile)
LSL = prctile(T.(continuousCTQ), 10);
USL = prctile(T.(continuousCTQ), 90);
fprintf('[INFO] Spec limits for %s: LSL=%.2f, USL=%.2f\n', continuousCTQ, LSL, USL);

%% 2) ANALYZE — DRIVERS & MODEL
fprintf('[PHASE 2] ANALYZE\n');

% Spearman correlation heatmap
corrMatrix = corr(T{:, featureNames}, 'type', 'Spearman');
fB = figure('Name','Figure B: Spearman Correlation Heatmap','NumberTitle','off','Visible','off');
heatmap(featureNames, featureNames, corrMatrix);
title('Figure B: Spearman Correlation of Features');
saveas(fB, fullfile(outDir,'Figure_B_Correlation_Heatmap.png'));
close(fB);

% Univariate screening (Mann-Whitney U / ranksum)
pValues = zeros(1, numel(featureNames));
for i = 1:numel(featureNames)
    f = featureNames{i};
    [p, ~] = ranksum(T.(f)(T.(targetFault)==1), T.(f)(T.(targetFault)==0));
    pValues(i) = p;
end

univariateResults = table(featureNames', pValues', ...
    'VariableNames', {'Feature','PValue'});
univariateResults = sortrows(univariateResults, 'PValue','ascend');

writetable(univariateResults, fullfile(outDir,'Table_A_Top_Features.csv'));

% Multivariate model (Logistic Regression) with holdout
shortlistK = min(cfg.steel_shortlist_k, height(univariateResults));
shortlistFeatures = univariateResults.Feature(1:shortlistK);
X = T(:, shortlistFeatures);
y = T.(targetFault);

cv     = cvpartition(y, 'HoldOut', cfg.steel_holdout); % deterministic due to rng(seed)
XTrain = X(cv.training, :);  yTrain = y(cv.training);
XTest  = X(cv.test, :);      yTest  = y(cv.test);

mdl = fitglm(XTrain, yTrain, 'Distribution','binomial', 'Link','logit');

% Evaluate
yPred_prob = predict(mdl, XTest);
yPred      = double(yPred_prob > cfg.steel_threshold);

% Confusion matrix
fC = figure('Name','Figure C: Confusion Matrix','NumberTitle','off','Visible','off');
cm = confusionchart(yTest, yPred);
cm.Title = 'Figure C: Confusion Matrix (Test Set)';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
saveas(fC, fullfile(outDir,'Figure_C_Confusion_Matrix.png'));
close(fC);

% ROC
[fx, fy, ~, AUC] = perfcurve(yTest, yPred_prob, 1);
fD = figure('Name','Figure D: ROC Curve','NumberTitle','off','Visible','off');
plot(fx, fy, 'LineWidth', 2); hold on; plot([0 1],[0 1],'r--'); hold off;
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(sprintf('Figure D: ROC Curve (AUC = %.2f)', AUC));
legend('Logistic Regression','Random Classifier','Location','southeast');
grid on;
saveas(fD, fullfile(outDir,'Figure_D_ROC_Curve.png'));
close(fD);

% Coefficients & odds ratios
modelCoefficients = mdl.Coefficients;
modelCoefficients.OddsRatio = exp(modelCoefficients.Estimate);
writetable(modelCoefficients, fullfile(outDir,'Table_B_Logistic_Regression_Results.csv'));

% Save a compact model summary
fid = fopen(fullfile(outDir,'Steel_Model_Summary.txt'),'w');
cleanupObj = onCleanup(@() fclose(fid));
fprintf(fid, "Logistic Regression on shortlist=%d\n", shortlistK);
fprintf(fid, "Predictors: %s\n\n", strjoin(shortlistFeatures', ', '));
fprintf(fid, "AUC: %.4f\n", AUC);
fprintf(fid, "Threshold: %.3f\n", cfg.steel_threshold);

clear cleanupObj;

%% 3) IMPROVE — DOE-Style Simulation & CAPABILITY
fprintf('[PHASE 3] IMPROVE\n');

significantPredictors = modelCoefficients( ...
    modelCoefficients.pValue < 0.05 & ~strcmp(modelCoefficients.Properties.RowNames,'(Intercept)'), :);
significantPredictors = sortrows(significantPredictors, 'pValue','ascend');

if height(significantPredictors) < 2
    error('Not enough significant predictors for DOE. Lower p-value threshold or broaden feature set.');
end

predictor1_name = significantPredictors.Properties.RowNames{1};
predictor2_name = significantPredictors.Properties.RowNames{2};
fprintf('[INFO] DOE on top predictors: %s, %s\n', predictor1_name, predictor2_name);

% Levels from class-conditional means
level_low_p1  = mean(T.(predictor1_name)(T.(targetFault)==1));
level_high_p1 = mean(T.(predictor1_name)(T.(targetFault)==0));
level_low_p2  = mean(T.(predictor2_name)(T.(targetFault)==1));
level_high_p2 = mean(T.(predictor2_name)(T.(targetFault)==0));

% Build DOE table with exact predictor names
doeTable = table();
doeTable.(predictor1_name) = [level_low_p1;  level_high_p1];
doeTable.(predictor2_name) = [level_low_p2;  level_high_p2];

% Fill remaining predictors with overall means, preserving names
otherFeatures = setdiff(mdl.PredictorNames, {predictor1_name, predictor2_name}, 'stable');
if ~isempty(otherFeatures)
    otherMeans = zeros(1, numel(otherFeatures));
    for k = 1:numel(otherFeatures)
        otherMeans(k) = mean(T.(otherFeatures{k}));
    end
    otherTable = array2table(repmat(otherMeans, height(doeTable), 1), 'VariableNames', otherFeatures);
    doeTable = [doeTable otherTable];
end
doeTable = doeTable(:, mdl.PredictorNames); % ensure order

predictedProb = predict(mdl, doeTable);
doeResults = table({'Current (Low Settings)'; 'Proposed (High Settings)'}, predictedProb, ...
    'VariableNames', {'Setting','PredictedDefectProbability'});
writetable(doeResults, fullfile(outDir,'Table_C_DOE_Simulation.csv'));

% Capability (Before vs After simulated)
dataBefore = T.(continuousCTQ);
[cp_before, cpk_before] = calculateCpk(dataBefore, USL, LSL);

dataAfter = T.(continuousCTQ)(T.(targetFault)==0); % simulate using non-faulty group
[cp_after, cpk_after] = calculateCpk(dataAfter, USL, LSL);

capabilityResults = table( ...
    {'Before'; 'After (Simulated)'}, ...
    [cp_before; cp_after], ...
    [cpk_before; cpk_after], ...
    'VariableNames', {'State','Cp','Cpk'});
writetable(capabilityResults, fullfile(outDir,'Table_D_Capability_Analysis.csv'));

%% 4) CONTROL — p-CHART & CONTROL PLAN
fprintf('[PHASE 4] CONTROL\n');

improvedDefectRate = doeResults.PredictedDefectProbability(2);
simulatedData = binornd(1, improvedDefectRate, height(T), 1); % deterministic due to rng(seed)

batchSize   = cfg.steel_batch_size;
numBatches  = floor(length(simulatedData) / batchSize);
proportions = zeros(1, numBatches);
for i = 1:numBatches
    batch = simulatedData((i-1)*batchSize+1 : i*batchSize);
    proportions(i) = sum(batch) / batchSize;
end

p_bar = mean(proportions);
UCL   = p_bar + 3 * sqrt(p_bar * (1 - p_bar) / batchSize);
LCL   = max(0, p_bar - 3 * sqrt(p_bar * (1 - p_bar) / batchSize));

fE = figure('Name','Figure E: p-Chart','NumberTitle','off','Visible','off');
plot(1:numBatches, proportions, '-o', 'LineWidth', 1.5, 'MarkerFaceColor','b'); hold on;
yline(p_bar, 'g--', 'LineWidth', 2, 'Label','Center Line');
yline(UCL,   'r--', 'LineWidth', 2, 'Label','UCL');
yline(LCL,   'r--', 'LineWidth', 2, 'Label','LCL');
hold off;
title('Figure E: p-Chart for Monitoring Improved Process');
xlabel('Batch Number'); ylabel('Proportion of Defects');
ylim([0, max(1e-6, UCL)*1.5]); grid on;
saveas(fE, fullfile(outDir,'Figure_E_p_Chart.png'));
close(fE);

% Control Plan (text)
planLines = [
    "- Sampling: Sample " + string(batchSize) + " plates per period; plot proportion of '" + string(targetFault) + "' on p-Chart."
    "- Reaction: If any point breaches control limits, halt and investigate for special causes."
    "- SOP: Update procedures to set control parameters for top predictors."
];
fid2 = fopen(fullfile(outDir,'Steel_Control_Plan.txt'),'w');
cleanupObj2 = onCleanup(@() fclose(fid2));
fprintf(fid2, "CONTROL PLAN — STEEL PLATES\n---------------------------\n");
fprintf(fid2, "%s\n", planLines);
clear cleanupObj2;

fprintf('\n[DONE] Steel Plates DMAIC complete. All outputs saved under: %s\n', outDir);

%% HELPER
function [Cp, Cpk] = calculateCpk(data, USL, LSL)
    mu = mean(data);
    sigma = std(data);
    if sigma == 0
        Cp = inf; Cpk = inf; return;
    end
    Cp  = (USL - LSL) / (6 * sigma);
    cpu = (USL - mu) / (3 * sigma);
    cpl = (mu - LSL) / (3 * sigma);
    Cpk = min(cpu, cpl);
end
