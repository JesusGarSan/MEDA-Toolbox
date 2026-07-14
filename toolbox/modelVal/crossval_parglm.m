function [T_crossval, parglmo_crossval] = crossval_parglm(kfolds, X, F, varargin)

% Cross-Validation Parallel General Linear Model for multivariate data analysis
% to estimate stability, mean effects, and standard errors of the 
% factorization matrix and interaction terms using k-fold partitioning.
%
% [T_crossval, parglmo_crossval] = crossval_parglm(kfolds, X, F)   % minimum call
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
% parglmo_crossval (structure): Structure returned from the baseline run on 
% the original data, containing the factor and interaction matrices.
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
% [T_crossval, parglmo_crossval] = crossval_parglm(5, X, F, 'Model', 'linear')
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
%% Main code
n = size(X, 1);

% Ensure kfolds is valid
if kfolds > n || kfolds < 2
    error('The number of folds must be between 2 and the number of observations (N).');
end

% Run baseline model on original data to dynamically capture current shape
[T_orig, parglmo_crossval] = parglm(X, F, varargin{:});

% Dynamically identify numerical columns for this specific execution
allVars = T_orig.Properties.VariableNames;
isNumericCol = cellfun(@(c) isnumeric(T_orig.(c)), allVars);
numericVars = allVars(isNumericCol);

numRows = height(T_orig);
numVars = length(numericVars);

% Allocate memory based on the dynamically discovered dimensions (kfolds sheets)
cvData = nan(numRows, numVars, kfolds);

% Randomly shuffle row indices at start for unbiased CV partitioning
shuffledIds = randperm(n);

% Generate fold indices (splitting as equally as possible)
foldSize = floor(n / kfolds);
remainder = rem(n, kfolds);

% Build block limits
limits = zeros(kfolds + 1, 1);
limits(1) = 1;
for i = 1:kfolds
    limits(i+1) = limits(i) + foldSize + (i <= remainder);
end

% Run cross-validation loop (Leave-One-Block-Out)
for i = 1:kfolds
    % Identify testing block indices
    testIdx = shuffledIds(limits(i) : (limits(i+1) - 1));
    
    % Training indices are all except the test block
    trainIdx = true(n, 1);
    trainIdx(testIdx) = false;
    
    % Fit model on training set (all folds except the current one)
    [T, ~] = parglm(X(trainIdx, :), F(trainIdx, :), varargin{:});
    
    % Safely populate the matrix slice
    for v = 1:numVars
        cvData(:, v, i) = T.(numericVars{v});
    end
end

% Compute mean and standard error across folds
meanVals = mean(cvData, 3, 'omitnan');
stdVals  = std(cvData, 0, 3, 'omitnan');
seVals   = stdVals ./ sqrt(kfolds); 

% Reconstruct T_crossval matching the baseline structure and column names
T_crossval = table();
T_crossval.Source = T_orig.Source; % Copy row descriptions directly

for v = 1:numVars
    colName = numericVars{v};
    
    % Preallocate a cell array for strings for this column
    formattedCol = cell(numRows, 1);
    
    for r = 1:numRows
        % If the original value was NaN (such as Residual P-values), preserve NaN
        if isnan(meanVals(r, v))
            formattedCol{r} = 'NaN';
        else
            % Format to 4 decimal places
            formattedCol{r} = sprintf('%.4f ± %.4f', meanVals(r, v), seVals(r, v));
        end
    end
    
    % Save with the exact original column name
    T_crossval.(colName) = formattedCol;
end

end