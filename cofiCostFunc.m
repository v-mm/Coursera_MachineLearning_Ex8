function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% 
% R is [num_movies x num_users]; R(i,j) = 1; => ith movie is rated by user j
% Y is [num_movies x num_users]; Y(i,j) = {0:5};=> rating of ith movie by user j
% i.e. Y is matrix of actual ratings
% X is [num_movies x num_features]
% Theta is [num_users x num_features] % basically each user has a certain
% weight for every feature in movie 'i'

% Y is matrix of actual ratings
predictedRatings = size(Y);
error = size(Y);

% [num_movies x num_features] *  [num_users x num_features]'
predictedRatings = X * Theta' % result is [num_movies x num_users]
% R is [num_movies x num_users]; element wise multiplication with R which has 
% 0 or 1 will consider only movies that have actual ratings
error = (predictedRatings - Y).*R; 

% unregularized cost
J = (1/2) * sum(sum(error.^2));

% X gradient = Sum over all users j who have rated (error * Theta)  
% [num_movies x num_users] * [num_users x num_features]
% result is [num_movies x num_features], same as X
X_grad = error * Theta;
% error matrix already accounts for R(i,j) = 1
% matrix multiplication inherently handles the summation

% similarly
% Theta gradient = Sum over all movies  i with ratings (error * X)
% [num_movies x num_users] * [num_movies x num_features]
% transposing to account for dimensions, we get 
% [num_users x num_movies] * [num_movies x num_features]
% result is [num_users x num_features], same as Theta
Theta_grad = error' * X;
% error matrix already accounts for R(i,j) = 1
% matrix multiplication inherently handles the summation

% see ex8.pdf for a different implementation of the above

% regularized cost (a.k.a cost at loaded parameters)
J = J + ((lambda/2) * sum(sum(Theta.^2))) + ((lambda/2) * sum(sum(X.^2)));

% regularized X gradient
% lambda is a scalar, X_grad and X have the same dimensions.
X_grad = X_grad + (lambda * X);

% regularized Theta gradient
% lambda is a scalar, Theta_grad and Theta have the same dimensions.
Theta_grad = Theta_grad + (lambda * Theta);

% =============================================================

grad = [X_grad(:); Theta_grad(:)];


end
