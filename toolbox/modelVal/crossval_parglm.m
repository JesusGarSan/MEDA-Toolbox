function [T_crossval, ascao_crossval] = crossval_parglm(kfolds, X, F, varargin)

% Cross-Validation Parallel General Linear Model for multivariate data analysis
% to estimate stability, mean effects, and standard errors of the 
% factorization matrix, interaction terms, loadings, scores, and residuals using k-fold partitioning.
%
% [T_crossval, ascao_crossval] = crossval_parglm(kfolds, X, F)   % minimum call
%
% See also: parglm, asca, apca, createDesign, boot_parglm
%
%
% INPUTS:
%
% kfolds: [1x1] number of folds (partitions) to split the dataset into 
% for cross-validation.
%
% X: [NxM] bilinear data set for model fitting, where each row is a
% measurement, each column a variable.
%
% F: [NxF] design matrix, cell or array, where columns correspond to 
% factors and rows to levels.
%
%
% Optional INPUTS (parameters):
%
% All optional parameters are passed to the underlying parglm
% function. These include:
%
% 'Model': Model type ('linear', 'interaction', 'full', or custom matrix/cell).
% 'Preprocessing': Preprocessing type (0: none, 1: mean-centering, 2: auto-scaling).
% 'Permutations': Number of permutations for significance testing within parglm.
% 'Ts': Test statistic selection (0: SSQ, 1: F-ratio, 2: Hierarchical F-ratio).
% 'Ordinal': Factor type array ([1xF] nominal/ordinal).
% 'Random': Factor variance array ([1xF] fixed/random).
% 'Fmtc': Multiple-test correction method (0 to 4).
% 'Coding': Factor coding scheme array ([1xF] sum/reference).
% 'Nested': Array mapping nested factor pairs.
% 'Type': Type of ANOVA factorization ('Simultaneous' or 'Sequential').
% 'Warning': Boolean flag to show preprocessing warnings.
% 'Parallel': Boolean flag to toggle parfor inside permutations.
%
%
% OUTPUTS:
%
% T_crossval (table): ANOVA-like summary table preserving the original parglm 
% column names. All numerical metrics are formatted as string columns 
% displaying 'Mean ± Standard Error' across the cross-validation folds.
%
% ascao_crossval (structure): Structure returned from the baseline ASCA model, 
% updated with cross-validated mean values and standard error matrices (.se)
% for loads, scoresV, and factor/model residuals.
%
%
% EXAMPLE OF USE (copy and paste the code in the command line)
%   Random data, two factors, 4 replicates, 5-fold cross-validation
%
% reps = 4;
% vars = 400;
% levels = {[1,2,3,4],[1,2,3]};
% 
% F = createDesign(levels,'Replicates',reps);
% 
% X = zeros(size(F,1),vars);
% for i = 1:length(levels{1}),
%     X(find(F(:,1) == levels{1}(i)),:) = simuleMV(length(find(F(:,1) == levels{1}(i))),vars,'LevelCorr',8) + repmat(randn(1,vars),length(find(F(:,1) == levels{1}(i))),1);
% end
% X = X + 100*ones(size(F,1),1)*rand(1,vars);
% 
% [T_crossval, ascao_crossval] = crossval_parglm(5, X, F, 'Model', 'linear')
%
%
% Coded by: Jesús García (gsus@ugr.es)
% Last modification: 14/Jul/2026
% Dependencies: Matlab R2024b, MEDA v1.13
%
% Copyright (C) 2026  University of Granada, Granada
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

%% Internal helper: Orthogonal Procrustes
function [r, yrot] = orthProc(x, y)
    % Computes orthogonal procrustes rotation matrix r projecting y onto x
    [u, ~, v] = svd(y' * x, 0);
    r = u * v';
    yrot = y * r;
end

%% Main code
n = size(X, 1);

% Ensure kfolds is valid
if kfolds > n || kfolds < 2
    error('The number of folds must be between 2 and the number of observations (N).');
end

% Run baseline model on original data to dynamically capture current shape
[T_orig, parglmo_orig] = parglm(X, F, varargin{:});
ascao_orig = asca(parglmo_orig);

% Initialize output structure from baseline model
ascao_crossval = ascao_orig;

% Dynamically identify numerical columns for this specific execution
allVars = T_orig.Properties.VariableNames;
isNumericCol = cellfun(@(c) isnumeric(T_orig.(c)), allVars);
numericVars = allVars(isNumericCol);

numRows = height(T_orig);
numVars = length(numericVars);

% Allocate memory for table summary metrics across folds
cvData = nan(numRows, numVars, kfolds);

% --- Procrustes & Factor/Interaction Preallocation ---
hasFactors = isfield(ascao_orig, 'factors') && ~isempty(ascao_orig.factors);
hasInters  = isfield(ascao_orig, 'interactions') && ~isempty(ascao_orig.interactions);

cvFactorLoads   = {};
cvFactorScoresV = {};
cvFactorRes     = {};

if hasFactors
    numFactors = length(ascao_orig.factors);
    cvFactorLoads   = cell(1, numFactors);
    cvFactorScoresV = cell(1, numFactors);
    cvFactorRes     = cell(1, numFactors);
    
    for f = 1:numFactors
        if isfield(ascao_orig.factors{f}, 'loads')
            [numVarsX, numComps] = size(ascao_orig.factors{f}.loads);
            cvFactorLoads{f} = nan(numVarsX, numComps, kfolds);
        end
        if isfield(ascao_orig.factors{f}, 'scoresV')
            [numObs, numComps] = size(ascao_orig.factors{f}.scoresV);
            cvFactorScoresV{f} = nan(numObs, numComps, kfolds);
        end
        if isfield(ascao_orig.factors{f}, 'residuals')
            [numObs, numVarsX] = size(ascao_orig.factors{f}.residuals);
            cvFactorRes{f} = nan(numObs, numVarsX, kfolds);
        end
    end
end

cvInterLoads   = {};
cvInterScoresV = {};
cvInterRes     = {};

if hasInters
    numInters = length(ascao_orig.interactions);
    cvInterLoads   = cell(1, numInters);
    cvInterScoresV = cell(1, numInters);
    cvInterRes     = cell(1, numInters);
    
    for i = 1:numInters
        if isfield(ascao_orig.interactions{i}, 'loads')
            [numVarsX, numComps] = size(ascao_orig.interactions{i}.loads);
            cvInterLoads{i} = nan(numVarsX, numComps, kfolds);
        end
        if isfield(ascao_orig.interactions{i}, 'scoresV')
            [numObs, numComps] = size(ascao_orig.interactions{i}.scoresV);
            cvInterScoresV{i} = nan(numObs, numComps, kfolds);
        end
        if isfield(ascao_orig.interactions{i}, 'residuals')
            [numObs, numVarsX] = size(ascao_orig.interactions{i}.residuals);
            cvInterRes{i} = nan(numObs, numVarsX, kfolds);
        end
    end
end

% Allocate global model residuals
cvModelRes = nan(size(X,1), size(X,2), kfolds);

% Randomly shuffle row indices at start for unbiased CV partitioning
shuffledIds = randperm(n);

% Generate fold indices
foldSize = floor(n / kfolds);
remainder = rem(n, kfolds);

limits = zeros(kfolds + 1, 1);
limits(1) = 1;
for i = 1:kfolds
    limits(i+1) = limits(i) + foldSize + (i <= remainder);
end

% Run cross-validation loop
for i = 1:kfolds
    testIdx = shuffledIds(limits(i) : (limits(i+1) - 1));
    trainIdx = true(n, 1);
    trainIdx(testIdx) = false;
    
    % Fit model on training set
    [T, parglmo] = parglm(X(trainIdx, :), F(trainIdx, :), varargin{:});
    ascao = asca(parglmo);

    % --- Procrustes Alignment & Factor Matrices Collection ---
    if hasFactors && isfield(ascao, 'factors')
        for f = 1:length(ascao_orig.factors)
            if isfield(ascao_orig.factors{f}, 'loads') && isfield(ascao.factors{f}, 'loads')
                X_target  = ascao_orig.factors{f}.loads;
                Y_current = ascao.factors{f}.loads;
                
                % Compute rotation matrix r
                [r, Y_rot] = orthProc(X_target, Y_current);
                cvFactorLoads{f}(:, :, i) = Y_rot;
                
                % Align scoresV using rotation matrix r
                if isfield(ascao.factors{f}, 'scoresV')
                    cvFactorScoresV{f}(trainIdx, :, i) = ascao.factors{f}.scoresV * r;
                end
            end
            
            % Record training residuals
            if isfield(ascao.factors{f}, 'residuals')
                cvFactorRes{f}(trainIdx, :, i) = ascao.factors{f}.residuals;
            end
        end
    end
    
    % --- Procrustes Alignment & Interaction Matrices Collection ---
    if hasInters && isfield(ascao, 'interactions')
        for f = 1:length(ascao_orig.interactions)
            if isfield(ascao_orig.interactions{f}, 'loads') && isfield(ascao.interactions{f}, 'loads')
                X_target  = ascao_orig.interactions{f}.loads;
                Y_current = ascao.interactions{f}.loads;
                
                [r, Y_rot] = orthProc(X_target, Y_current);
                cvInterLoads{f}(:, :, i) = Y_rot;
                
                if isfield(ascao.interactions{f}, 'scoresV')
                    cvInterScoresV{f}(trainIdx, :, i) = ascao.interactions{f}.scoresV * r;
                end
            end
            
            if isfield(ascao.interactions{f}, 'residuals')
                cvInterRes{f}(trainIdx, :, i) = ascao.interactions{f}.residuals;
            end
        end
    end
    
    % Collect model residuals
    if isfield(ascao, 'residuals')
        cvModelRes(trainIdx, :, i) = ascao.residuals;
    end
    
    % Populate ANOVA metrics table slice
    for v = 1:numVars
        cvData(:, v, i) = T.(numericVars{v});
    end
end

% --- Compute Mean and SE for ANOVA Summary Table ---
meanVals = mean(cvData, 3, 'omitnan');
stdVals  = std(cvData, 0, 3, 'omitnan');
seVals   = stdVals ./ sqrt(kfolds); 

T_crossval = table();
T_crossval.Source = T_orig.Source;

for v = 1:numVars
    colName = numericVars{v};
    formattedCol = cell(numRows, 1);
    
    for r = 1:numRows
        if isnan(meanVals(r, v))
            formattedCol{r} = 'NaN';
        else
            formattedCol{r} = sprintf('%.4f ± %.4f', meanVals(r, v), seVals(r, v));
        end
    end
    T_crossval.(colName) = formattedCol;
end

% --- Populate Cross-Validated Metrics back into ascao_crossval ---

% Global Model Residuals
if isfield(ascao_orig, 'residuals')
    ascao_crossval.residuals    = mean(cvModelRes, 3, 'omitnan');
    ascao_crossval.residuals_se = std(cvModelRes, 0, 3, 'omitnan') ./ sqrt(kfolds);
end

% Factors
if hasFactors
    for f = 1:numFactors
        if isfield(ascao_orig.factors{f}, 'loads')
            ascao_crossval.factors{f}.loads    = mean(cvFactorLoads{f}, 3, 'omitnan');
            ascao_crossval.factors{f}.loads_se = std(cvFactorLoads{f}, 0, 3, 'omitnan') ./ sqrt(kfolds);
        end
        if isfield(ascao_orig.factors{f}, 'scoresV')
            ascao_crossval.factors{f}.scoresV    = mean(cvFactorScoresV{f}, 3, 'omitnan');
            ascao_crossval.factors{f}.scoresV_se = std(cvFactorScoresV{f}, 0, 3, 'omitnan') ./ sqrt(kfolds);
        end
        if isfield(ascao_orig.factors{f}, 'residuals')
            ascao_crossval.factors{f}.residuals    = mean(cvFactorRes{f}, 3, 'omitnan');
            ascao_crossval.factors{f}.residuals_se = std(cvFactorRes{f}, 0, 3, 'omitnan') ./ sqrt(kfolds);
        end
    end
end

% Interactions
if hasInters
    for i = 1:numInters
        if isfield(ascao_orig.interactions{i}, 'loads')
            ascao_crossval.interactions{i}.loads    = mean(cvInterLoads{i}, 3, 'omitnan');
            ascao_crossval.interactions{i}.loads_se = std(cvInterLoads{i}, 0, 3, 'omitnan') ./ sqrt(kfolds);
        end
        if isfield(ascao_orig.interactions{i}, 'scoresV')
            ascao_crossval.interactions{i}.scoresV    = mean(cvInterScoresV{i}, 3, 'omitnan');
            ascao_crossval.interactions{i}.scoresV_se = std(cvInterScoresV{i}, 0, 3, 'omitnan') ./ sqrt(kfolds);
        end
        if isfield(ascao_orig.interactions{i}, 'residuals')
            ascao_crossval.interactions{i}.residuals    = mean(cvInterRes{i}, 3, 'omitnan');
            ascao_crossval.interactions{i}.residuals_se = std(cvInterRes{i}, 0, 3, 'omitnan') ./ sqrt(kfolds);
        end
    end
end

end