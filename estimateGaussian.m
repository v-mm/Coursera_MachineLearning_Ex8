function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%


muRow = mean(X);
% muRow is a n element row vector of means of the n columns of X
mu = muRow';
% mu is a n x 1 column vector

sigma2Row = (1/m) * sum((bsxfun(@minus, X, muRow)).^2);
% sigma2Row is a n element row vector of variance of the n columns of X
sigma2 = sigma2Row';
% sigma2 is a n x 1 column vector

% =============================================================


end
