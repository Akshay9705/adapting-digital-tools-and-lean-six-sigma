%% DMAIC CASE STUDY 2: NASA C-MAPSS PREDICTIVE MAINTENANCE & RESILIENCE
%
% Author: Akshay Thummalapally
% Date: August 2025
%
% Description:
% Complete DMAIC (Define, Measure, Analyze, Improve, Control) cycle on the
% C-MAPSS FD001 dataset to demonstrate the framework for predictive
% maintenance (Industry 5.0 pillar: Resilience).
%
% ONE-TIME SETUP:
% - Place this script and these files in the same folder:
%   train_FD001.txt, test_FD001.txt, RUL_FD001.txt
% - In MATLAB, cd to that folder and run.
%
% TOOLBOXES REQUIRED:
% - Statistics and Machine Learning Toolbox
% - (Optional) Predictive Maintenance Toolbox for advanced features

%% 0) SETUP & DATA LOADING
clear; clc; close all;

% Column names for C-MAPSS (FD001 has 26 columns: 1..26)
colNames = {'UnitNumber','TimeCycle','OpSet1','OpSet2','OpSet3',...
    'Sensor1','Sensor2','Sensor3','Sensor4','Sensor5','Sensor6','Sensor7',...
    'Sensor8','Sensor9','Sensor10','Sensor11','Sensor12','Sensor13',...
    'Sensor14','Sensor15','Sensor16','Sensor17','Sensor18','Sensor19',...
    'Sensor20','Sensor21'};

try
    trainData = readtable('train_FD001.txt','FileType','text','Delimiter',' ');
    trainData.Properties.VariableNames = colNames;

    testData  = readtable('test_FD001.txt','FileType','text','Delimiter',' ');
    testData.Properties.VariableNames = colNames;

    trueRUL   = readtable('RUL_FD001.txt','FileType','text');
    trueRUL.Properties.VariableNames = {'TrueRUL'};

    disp('All datasets (train_FD001, test_FD001, RUL_FD001) loaded successfully.');
catch ME
    error('Data load failed: %s\nMake sure all three files are in the same folder as this script.', ME.message);
end

disp('--------------------------------------------------');

%% 1) DEFINE & MEASURE — BASELINE
disp('PHASE 1: DEFINE & MEASURE');

% RUL for training data (runs to failure)
maxCycles = grpstats(trainData.TimeCycle, trainData.UnitNumber, 'max');
maxCyclesTable = table(unique(trainData.UnitNumber), maxCycles, ...
    'VariableNames', {'UnitNumber','MaxCycle'});
trainData = outerjoin(trainData, maxCyclesTable, 'Keys','UnitNumber','MergeKeys',true);
trainData.RUL = trainData.MaxCycle - trainData.TimeCycle;
trainData.MaxCycle = [];
disp('Calculated RUL for training data.');

% Visualise representative sensor degradation (Sensors 7 & 12 for engines 1..5)
figure('Name','Figure A: Representative Sensor Degradation','NumberTitle','off');
subplot(2,1,1); hold on;
for i = 1:5
    ed = trainData(trainData.UnitNumber==i,:);
    plot(ed.TimeCycle, ed.Sensor7);
end
hold off; grid on;
title('Figure A: Sensor 7 Trajectories');
xlabel('Time (Cycles)'); ylabel('Sensor 7 Reading');

subplot(2,1,2); hold on;
for i = 1:5
    ed = trainData(trainData.UnitNumber==i,:);
    plot(ed.TimeCycle, ed.Sensor12);
end
hold off; grid on;
title('Sensor 12 Trajectories');
xlabel('Time (Cycles)'); ylabel('Sensor 12 Reading');
saveas(gcf,'Figure_A_Sensor_Degradation.png');
disp('Figure A (Sensor Degradation) saved as PNG.');

% Baseline failure-time distribution (training set)
failureTimes = maxCyclesTable.MaxCycle;
figure('Name','Figure B: Histogram of Engine Failure Times','NumberTitle','off');
histogram(failureTimes,15); grid on;
title('Figure B: Baseline Distribution of Engine Failure Times');
xlabel('Time to Failure (Cycles)'); ylabel('Frequency');
saveas(gcf,'Figure_B_Failure_Histogram.png');
baselineMTBF = mean(failureTimes);
fprintf('Baseline Mean Time Between Failures (MTBF): %.2f cycles\n', baselineMTBF);

disp('--------------------------------------------------');

%% 2) ANALYZE — IDENTIFY PREDICTIVE SIGNALS
disp('PHASE 2: ANALYZE');

% Remove near-constant sensors (no predictive value). Keep OpSets.
sensorVars = trainData.Properties.VariableNames(6:26); % Sensor1..Sensor21
sensorStd  = std(trainData{:,sensorVars});
constantSensors = sensorVars(sensorStd < 1e-2); % threshold can be tuned
if ~isempty(constantSensors)
    trainData(:,constantSensors) = [];
    testData(:,constantSensors)  = [];
end

% Refresh list of predictors after removal (keep OpSets 1-3)
predictorVars = trainData.Properties.VariableNames(3:width(trainData)-1); % OpSet1..OpSet3 + remaining sensors (exclude RUL at end)

% Spearman correlation to RUL (training set)
corrWithRUL = zeros(numel(predictorVars),1);
for i = 1:numel(predictorVars)
    corrWithRUL(i) = corr(trainData.(predictorVars{i}), trainData.RUL,'type','Spearman','rows','complete');
end
corrTable = table(predictorVars', abs(corrWithRUL), 'VariableNames',{'Feature','AbsSpearmanRUL'});
corrTable = sortrows(corrTable, 'AbsSpearmanRUL','descend');

disp('Table A: Top Feature Correlations with RUL (absolute Spearman)');
disp(corrTable(1:min(12,height(corrTable)),:));
writetable(corrTable,'Table_A_Sensor_Correlation.csv');
disp('Table A (Sensor Correlation) saved as CSV.');

% Choose top predictors (e.g., top 8)
kTop = min(8, height(corrTable));
topPredictors = corrTable.Feature(1:kTop)';

% Fit a simple regression model: RUL ~ top predictors (baseline model)
formula = ['RUL ~ ' strjoin(topPredictors,' + ')];
rulModel = fitlm(trainData, formula);
disp('Linear Regression Model for RUL:');
disp(rulModel);

% ----- Get LAST cycle per engine in TEST set (robust, version-agnostic) -----
testSorted = sortrows(testData, {'UnitNumber','TimeCycle'});
isLastRowOfGroup = [diff(testSorted.UnitNumber)~=0; true];
lastCycleTestData = testSorted(isLastRowOfGroup,:);
lastCycleTestData = sortrows(lastCycleTestData,'UnitNumber');

% Sanity-check alignment with RUL rows
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
fprintf('Model Performance on Test Set: RMSE = %.2f, MAE = %.2f\n', RMSE, MAE);

% Predicted vs Actual plot
figure('Name','Figure C: Predicted vs. Actual RUL','NumberTitle','off');
scatter(trueRUL.TrueRUL, predictedRUL, 'filled'); hold on;
mx = max([trueRUL.TrueRUL; predictedRUL]);
plot([0 mx],[0 mx],'r--','LineWidth',2);
grid on; hold off;
title(sprintf('Figure C: Predicted vs. Actual RUL (RMSE = %.2f)', RMSE));
xlabel('True RUL (Cycles)'); ylabel('Predicted RUL (Cycles)');
legend('Predictions','Perfect Prediction','Location','northwest');
saveas(gcf,'Figure_C_Predicted_vs_Actual.png');
disp('Figure C (Predicted vs. Actual RUL) saved as PNG.');

disp('--------------------------------------------------');

%% 3) IMPROVE — MAINTENANCE POLICY (PdM TRIGGER)
disp('PHASE 3: IMPROVE');

% Simple policy: maintenance if predicted RUL < threshold
maintenanceThreshold = 30; % cycles
fprintf('Maintenance Policy: Replace component if predicted RUL < %d cycles.\n', maintenanceThreshold);

% Without PdM, assume all failures are unplanned in test horizon
unplanned_before = height(trueRUL);

% Missed detections = true RUL < threshold BUT predicted >= threshold
missed = sum(trueRUL.TrueRUL < maintenanceThreshold & predictedRUL >= maintenanceThreshold);
unplanned_after = missed;

downtimeReduction = (unplanned_before - unplanned_after) / unplanned_before * 100;
fprintf('Unplanned Failures — Before: %d | After: %d | Reduction: %.2f%%\n', ...
    unplanned_before, unplanned_after, downtimeReduction);

% Weibull fit to baseline failure times (training)
pd = fitdist(failureTimes, 'Weibull');
beta_before = pd.B;  % shape
eta_before  = pd.A;  % scale (characteristic life)
fprintf('Baseline Weibull: beta = %.2f, eta = %.2f\n', beta_before, eta_before);

% Illustrative improvement: increase characteristic life by 15%
eta_after = eta_before * 1.15;
mtbf_before = eta_before * gamma(1 + 1/beta_before);
mtbf_after  = eta_after  * gamma(1 + 1/beta_before);

% ---- FIXED: make table columns the same height (2×1) ----
State = ["Baseline"; "After PdM"];        % 2×1 string array
MTBF_Cycles = [mtbf_before; mtbf_after];  % 2×1 double
resilienceResults = table(State, MTBF_Cycles);

disp('Table B: Resilience Improvement (MTBF)');
disp(resilienceResults);
writetable(resilienceResults,'Table_B_Resilience_Improvement.csv');
disp('Table B (Resilience Improvement) saved as CSV.');

disp('--------------------------------------------------');

%% 4) CONTROL — MONITORING & SPC
disp('PHASE 4: CONTROL');

% Residual SPC (stability of prediction error)
residuals = trueRUL.TrueRUL - predictedRUL;
figure('Name','Figure D: SPC Chart of Model Residuals','NumberTitle','off');
controlchart(residuals,'charttype','i');
title('Figure D: Control Chart of RUL Prediction Residuals');
xlabel('Engine Unit (Test Set)'); ylabel('Prediction Error (Cycles)');
saveas(gcf,'Figure_D_SPC_Residuals.png');
disp('Figure D (SPC Chart) saved as PNG.');

% Control plan (textual)
disp('Control Plan Outline:');
disp('- Monitor RUL predictions each cycle; alert if Predicted RUL < 30 cycles.');
disp('- Review SPC chart of residuals quarterly; retrain if out-of-control.');
disp('- Provide dashboard: Predicted RUL vs Time per engine for operator support.');

disp('--------------------------------------------------');
disp('DMAIC ANALYSIS COMPLETE. All figures and tables saved in the current folder.');
