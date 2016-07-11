function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta0 = ones(size(X, 1), 1);
X_prime = [Theta0 X];
h_prime = X_prime * Theta1';
h_prime_exp = exp(-h_prime);
h_prime_add = 1.+h_prime_exp;
h_int = 1./h_prime_add;
h_int_1 = [Theta0 h_int];
y_prime = h_int_1 * Theta2';
y_prime_exp = exp(-y_prime);
y_prime_add = 1.+y_prime_exp;
h = 1./y_prime_add;

y_new = (y == 1);
for i = 2:num_labels,
  y_vec = (y == i);
  y_new = [y_new y_vec];
end;

h_pos = log(h);
h_neg = 1.-h;
h_neg = log(h_neg);
y_neg = 1.-y_new;
mult = ((y_new' * h_pos) + (y_neg' * h_neg));
length = size(mult, 1);
I = eye(length);
mul = I .* mult; %Take the diagonal elements
total = sum(mul);
total = sum(total);
J = -1 * (total/m);

T1 = Theta1(:,2:end);
T2 = Theta2(:,2:end);
T1 = T1.*T1;
T2 = T2.*T2;
T1_vec = sum(T1);
T2_vec = sum(T2);
T1_total = sum(T1_vec);
T2_total = sum(T2_vec);
T = T1_total + T2_total;
Reg = (T * lambda) / (2 * m);
J = J + Reg;

% Backpropagation
D1_grad = zeros(size(Theta1));
D2_grad = zeros(size(Theta2));
for t = 1:m,
  a_1 = X_prime(t,:);
  z_2 = a_1 * Theta1';
  z_2_exp = exp(-z_2);
  z_2_add = 1.+z_2_exp;
  z_2_int = 1./z_2_add;
  a_2 = [1 z_2_int];
  z_3 = a_2 * Theta2';
  z_3_exp = exp(-z_3);
  z_3_add = 1.+z_3_exp;
  a_3 = 1./z_3_add;
  y_3 = (y_new(t,:));
  delta_3 = a_3 .- y_3;
  err_2 = delta_3 * Theta2;
  z_2_ext = [1 z_2];
  g_prime = sigmoidGradient(z_2_ext);
  delta_2 = err_2 .* g_prime;
  D2_a = delta_3' * a_2;
  D2_grad = D2_grad .+ D2_a; 
  delta_2 = delta_2(:,2:end);
  D1_a = delta_2' * a_1;
  D1_grad = D1_grad .+ D1_a;
end;

Theta2_grad = D2_grad ./ m;
Theta1_grad = D1_grad ./ m;

reg = lambda / m;

Theta1_reg = zeros(size(Theta1));
Theta2_reg = zeros(size(Theta2));

Theta1_reg(:,2:end) = Theta1(:,2:end);
Theta2_reg(:,2:end) = Theta2(:,2:end);

Theta1_reg = Theta1_reg .* reg; 
Theta2_reg = Theta2_reg .* reg; 

Theta1_grad = Theta1_grad .+ Theta1_reg;
Theta2_grad = Theta2_grad .+ Theta2_reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
