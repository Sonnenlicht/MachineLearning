function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
z = X * theta; 
h = sigmoid(z);
a = log(h);
b = 1.-h;
b = log(b); %log(1-h)
k = 1.-y;
delta = (-y' * a) - (k' * b);
J = delta / m;
sqrtheta = theta.^2;
sqrtheta(1) = 0;
reg = lambda * sum(sqrtheta) / (2*m);
J = J + reg;
diff = h - y;
grad = (X' * diff)./m;
%grad = grad .+ (lambda * theta)./m;
for l = 2:size(theta),
  grad(l) = grad(l) .+ (lambda * theta(l))./m; 

% =============================================================

end
