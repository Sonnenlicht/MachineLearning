function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta;
err = h.-y;
sqr_err = err.^2;
lin = sum(sqr_err) / (2*m);
sqr_theta = theta.^2;
sqr_theta(1) = 0;
reg = (lambda .* sqr_theta) ./ (2*m);
reg = sum(reg);
J = lin + reg;

reg = (lambda .* theta) ./ m;
reg(1) = 0;
grad_init = (X' * err) ./ m;
grad = grad_init .+ reg;

% =========================================================================

grad = grad(:);

end
