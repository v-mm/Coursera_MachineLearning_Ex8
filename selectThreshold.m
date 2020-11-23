function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions


% ==================================
% my notes:
% randomly initialize epsilon to smallest pval
% for this value of epsilon we compute F1 score.
% if computed F1 score is higher than previous F1, save this as bestF1 score
% also save epsilon that gives bestF1 (higher F1 score means better accuracy)
% increment epsilon with small step size and repeat process steps until 
% epsilon is max pval value

% F1 score = 2 * (precision * recall) / (precision + recall)
% precision = truePositives / (tuePositives + falsePositives)
% recall = truePositives / (tuePositives + falseNegatives)

% yval == 1, => actual anomaly
% pval < epsilon, => predicted anomaly

% my notes end
% ==================================

    truePositives = 0;
    falsePositives = 0;
    falseNegatives = 0;
    precision = 0;
    recall = 0;

    % count of truePositives - prediction and actual are both positive
    truePositives = sum((pval < epsilon)&(yval == 1));

    % count of falsePositives - prediction is positive but actual is negative
    falsePositives = sum((pval < epsilon)&(yval == 0));

    % count of falseNegatives - prediction is negative but actual is positive
    falseNegatives = sum((pval > epsilon)&(yval == 1));

    precision = truePositives / (truePositives + falsePositives);
    recall = truePositives / (truePositives + falseNegatives);

    F1 = 2 * (precision * recall)/(precision + recall);

    % =============================================================
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end

end

end
