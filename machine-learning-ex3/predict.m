function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

Theta0 = ones(size(X, 1), 1);
X_prime = [Theta0 X];
h_prime = X_prime * Theta1';
h_prime_exp = exp(-h_prime);
h_prime_add = 1.+h_prime_exp;
h = 1./h_prime_add;
h = [Theta0 h];
y_prime = h * Theta2';
y_prime_exp = exp(-y_prime);
y_prime_add = 1.+y_prime_exp;
y = 1./y_prime_add;

[high, p] = max(y, [], 2);
% =========================================================================


end
