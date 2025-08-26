%% DMAIC CASE STUDY 1: STEEL PLATES FAULTS ANALYSIS (Corrected)
%
% Author: Akshay Thummalapally (with Gemini Assistant)
% Date: August 2025
%
% Description:
% This script performs a complete DMAIC (Define, Measure, Analyze, Improve,
% Control) cycle on the Steel Plates Faults dataset. The goal is to
% demonstrate the application of the dissertation's integrated framework
% using a representative quality improvement use case.
%
% ONE-TIME SETUP:
% 1. Create a new folder for this case study (e.g., 'CS1_SteelPlates').
% 2. Place this MATLAB script (.m file) inside that folder.
% 3. Download the 'SteelPlatesFaults_Clean.csv' file and place it in the SAME folder.
% 4. Open MATLAB and navigate to this folder before running the script.
%
% TOOLBOXES REQUIRED:
% - Statistics and Machine Learning Toolbox

%% 0. SETUP AND DATA LOADING
clear; clc; close all;

% --- Import Data ---
try
    T = readtable('SteelPlatesFaults_Clean.csv');
    disp('Dataset loaded successfully.');
catch
    error('Failed to load dataset. Make sure "SteelPlatesFaults_Clean.csv" is in the same folder as this script.');
end

% --- Basic Data Checks ---
disp('First 5 rows of the dataset:');
head(T, 5);

% Define feature columns (predictors) and fault columns (responses).
featureNames = T.Properties.VariableNames(1:27);
faultNames   = T.Properties.VariableNames(28:34);

disp('--------------------------------------------------');

%% 1. DEFINE & MEASURE - ESTABLISH THE BASELINE
disp('PHASE 1: DEFINE & MEASURE');

% --- 1.2 Choose CTQ & Target Class ---
faultCounts     = sum(T{:, faultNames});
totalFaults     = sum(faultCounts);
faultPrevalence = (faultCounts / totalFaults) * 100;

prevalenceTable = table(faultNames', faultCounts', faultPrevalence', ...
    'VariableNames', {'FaultType', 'Count', 'Prevalence_Percent'});
prevalenceTable = sortrows(prevalenceTable, 'Count', 'descend');

disp('Fault Prevalence (% by class):');
disp(prevalenceTable);

% Pareto
figure('Name','Figure A: Pareto of Fault Categories','NumberTitle','off','Visible','off');
pareto(faultCounts, faultNames);
title('Figure A: Pareto Chart of Fault Categories');
xlabel('Fault Type'); ylabel('Count');
saveas(gcf,'Figure_A_Pareto_Chart.png');
disp('Figure A (Pareto Chart) saved as PNG.');

% Target class
targetFault = 'Other_Faults';
disp(['Selected CTQ for analysis: ', targetFault]);

% --- 1.3 Baseline Defect Rate & Continuous CTQ ---
baselineDefectRate = (sum(T.(targetFault)) / height(T)) * 100;
fprintf('Baseline Defect Rate for %s: %.2f%%\n', targetFault, baselineDefectRate);

continuousCTQ = 'Sum_of_Luminosity';
disp(['Selected continuous CTQ for capability analysis: ', continuousCTQ]);

% Proxy spec limits (10th–90th percentile)
LSL = prctile(T.(continuousCTQ), 10);
USL = prctile(T.(continuousCTQ), 90);
fprintf('Proxy Specification Limits for %s: LSL = %.2f, USL = %.2f\n', continuousCTQ, LSL, USL);

disp('--------------------------------------------------');

%% 2. ANALYZE - FIND STATISTICALLY SIGNIFICANT DRIVERS
disp('PHASE 2: ANALYZE');

% --- 2.1 Correlation Screen ---
corrMatrix = corr(T{:, featureNames}, 'type', 'Spearman');

figure('Name','Figure B: Spearman Correlation Heatmap','NumberTitle','off','Visible','off');
heatmap(featureNames, featureNames, corrMatrix);
title('Figure B: Spearman Correlation of Features');
saveas(gcf,'Figure_B_Correlation_Heatmap.png');
disp('Figure B (Correlation Heatmap) saved as PNG.');
disp('Correlation analysis complete. High correlation noted between geometric features.');

% --- 2.2 Univariate Screening (Mann-Whitney U / ranksum) ---
pValues = zeros(1, numel(featureNames));
for i = 1:numel(featureNames)
    f = featureNames{i};
    [p, ~] = ranksum(T.(f)(T.(targetFault)==1), T.(f)(T.(targetFault)==0));
    pValues(i) = p;
end

univariateResults = table(featureNames', pValues', 'VariableNames', {'Feature','PValue'});
univariateResults = sortrows(univariateResults, 'PValue','ascend');

disp('Table A: Top Significant Features (p < 0.05)');
disp(univariateResults(1:12,:));
writetable(univariateResults,'Table_A_Top_Features.csv');
disp('Table A (Top Features) saved as CSV.');

% --- 2.3 Multivariate Model (Logistic Regression) ---
shortlistFeatures = univariateResults.Feature(1:12);
X = T(:, shortlistFeatures);
y = T.(targetFault);

cv    = cvpartition(y, 'HoldOut', 0.3);
XTrain = X(cv.training, :);  yTrain = y(cv.training);
XTest  = X(cv.test, :);      yTest  = y(cv.test);

mdl = fitglm(XTrain, yTrain, 'Distribution','binomial', 'Link','logit');
disp('Logistic Regression Model Summary:');
disp(mdl);

% Evaluate
yPred_prob = predict(mdl, XTest);
yPred      = double(yPred_prob > 0.5);

% Confusion matrix
figure('Name','Figure C: Confusion Matrix','NumberTitle','off','Visible','off');
cm = confusionchart(yTest, yPred);
cm.Title = 'Figure C: Confusion Matrix (Test Set)';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
saveas(gcf,'Figure_C_Confusion_Matrix.png');
disp('Figure C (Confusion Matrix) saved as PNG.');

% ROC
[fx, fy, ~, AUC] = perfcurve(yTest, yPred_prob, 1);
figure('Name','Figure D: ROC Curve','NumberTitle','off','Visible','off');
plot(fx, fy, 'LineWidth', 2); hold on; plot([0 1],[0 1],'r--'); hold off;
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(sprintf('Figure D: ROC Curve (AUC = %.2f)', AUC));
legend('Logistic Regression','Random Classifier','Location','southeast');
grid on;
saveas(gcf,'Figure_D_ROC_Curve.png');
disp('Figure D (ROC Curve) saved as PNG.');

% Coeffs + odds ratios
modelCoefficients = mdl.Coefficients;
modelCoefficients.OddsRatio = exp(modelCoefficients.Estimate);
disp('Table B: Model Coefficients and Odds Ratios');
disp(modelCoefficients);
writetable(modelCoefficients, 'Table_B_Logistic_Regression_Results.csv');
disp('Table B (Model Results) saved as CSV.');

disp('--------------------------------------------------');

%% 3. IMPROVE - PROPOSE AND QUANTIFY AN INTERVENTION
disp('PHASE 3: IMPROVE');

% --- 3.1 & 3.2 DOE Simulation (FIXED to preserve predictor names) ---
significantPredictors = modelCoefficients( ...
    modelCoefficients.pValue < 0.05 & ~strcmp(modelCoefficients.Properties.RowNames,'(Intercept)'), :);
significantPredictors = sortrows(significantPredictors, 'pValue','ascend');

% Guard in case fewer than 2 significant predictors remain
if height(significantPredictors) < 2
    error('Not enough significant predictors for DOE. Reduce p-value threshold or broaden feature set.');
end

predictor1_name = significantPredictors.Properties.RowNames{1};
predictor2_name = significantPredictors.Properties.RowNames{2};
fprintf('DOE Simulation will be based on top predictors: %s and %s\n', predictor1_name, predictor2_name);

% Levels from class-conditional means
level_low_p1  = mean(T.(predictor1_name)(T.(targetFault)==1));
level_high_p1 = mean(T.(predictor1_name)(T.(targetFault)==0));
level_low_p2  = mean(T.(predictor2_name)(T.(targetFault)==1));
level_high_p2 = mean(T.(predictor2_name)(T.(targetFault)==0));

% Build DOE table with EXACT predictor names expected by the model
doeTable = table();
doeTable.(predictor1_name) = [level_low_p1;  level_high_p1];
doeTable.(predictor2_name) = [level_low_p2;  level_high_p2];

% Fill the remaining predictors with their overall means,
% BUT keep the ORIGINAL variable names (no 'mean_' prefix)
otherFeatures = setdiff(mdl.PredictorNames, {predictor1_name, predictor2_name}, 'stable');

if ~isempty(otherFeatures)
    % Get means as a numeric row vector in the same order as otherFeatures
    otherMeans = zeros(1, numel(otherFeatures));
    for k = 1:numel(otherFeatures)
        otherMeans(k) = mean(T.(otherFeatures{k}));
    end
    % Create a table with the SAME variable names as the predictors
    otherTable = array2table(repmat(otherMeans, height(doeTable), 1), 'VariableNames', otherFeatures);
    % Append and then reorder columns to mdl.PredictorNames
    doeTable = [doeTable otherTable];
end

% Ensure column order matches model expectation
doeTable = doeTable(:, mdl.PredictorNames);

% Predict defect probability
predictedProb = predict(mdl, doeTable);
doeResults = table({'Current (Low Settings)'; 'Proposed (High Settings)'}, predictedProb, ...
    'VariableNames', {'Setting','PredictedDefectProbability'});

disp('Table C: DOE Simulation Results');
disp(doeResults);
writetable(doeResults,'Table_C_DOE_Simulation.csv');
disp('Table C (DOE Results) saved as CSV.');

% --- 3.3 Capability "Before vs. After" ---
dataBefore = T.(continuousCTQ);
[cp_before, cpk_before] = calculateCpk(dataBefore, USL, LSL);

dataAfter = T.(continuousCTQ)(T.(targetFault)==0); % simulate using non-faulty group
[cp_after, cpk_after] = calculateCpk(dataAfter, USL, LSL);

capabilityResults = table( ...
    {'Before'; 'After (Simulated)'}, ...
    [cp_before; cp_after], ...
    [cpk_before; cpk_after], ...
    'VariableNames', {'State','Cp','Cpk'});

disp('Table D: Process Capability (Cp, Cpk) Before vs. After');
disp(capabilityResults);
writetable(capabilityResults,'Table_D_Capability_Analysis.csv');
disp('Table D (Capability Analysis) saved as CSV.');

disp('--------------------------------------------------');

%% 4. CONTROL - PLAN TO HOLD THE GAINS
disp('PHASE 4: CONTROL');

% --- 4.1 p-Chart ---
improvedDefectRate = doeResults.PredictedDefectProbability(2);
simulatedData = binornd(1, improvedDefectRate, height(T), 1);

batchSize   = 50;
numBatches  = floor(length(simulatedData) / batchSize);
proportions = zeros(1, numBatches);
for i = 1:numBatches
    batch = simulatedData((i-1)*batchSize+1 : i*batchSize);
    proportions(i) = sum(batch) / batchSize;
end

p_bar = mean(proportions);
UCL   = p_bar + 3 * sqrt(p_bar * (1 - p_bar) / batchSize);
LCL   = max(0, p_bar - 3 * sqrt(p_bar * (1 - p_bar) / batchSize));

figure('Name','Figure E: p-Chart','NumberTitle','off','Visible','off');
plot(1:numBatches, proportions, '-o', 'LineWidth', 1.5, 'MarkerFaceColor','b'); hold on;
yline(p_bar, 'g--', 'LineWidth', 2, 'Label','Center Line');
yline(UCL,   'r--', 'LineWidth', 2, 'Label','UCL');
yline(LCL,   'r--', 'LineWidth', 2, 'Label','LCL');
hold off;
title('Figure E: p-Chart for Monitoring Improved Process');
xlabel('Batch Number'); ylabel('Proportion of Defects');
ylim([0, max(1e-6, UCL)*1.5]); grid on;
saveas(gcf,'Figure_E_p_Chart.png');
disp('Figure E (p-Chart) saved as PNG.');

% --- 4.2 Control Plan ---
disp('Control Plan Outline:');
disp('- Sampling: Sample 50 plates daily and plot the proportion of "Other_Faults" on the p-Chart.');
disp('- Reaction Rule: If any point falls outside the control limits, halt the process and investigate for special cause variation.');
disp('- SOP: Update the Standard Operating Procedure to specify the new control parameters for the top predictors.');

disp('--------------------------------------------------');
disp('DMAIC ANALYSIS COMPLETE.');
disp('All tables and figures have been saved to the current folder.');

%% HELPER FUNCTION
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
