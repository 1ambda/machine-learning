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

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

k = num_labels;

# initialize Y as the classification output
diagonal = eye(k);
Y = diagonal(y, :);

## Y = zeros(m, 10); 
## for i = 1:10
##     Y(:, i) = (y == i);
## endfor

# forward propagation to get h(x)

bias = ones(m, 1);
a1 = [bias X];

z2 = a1 * Theta1';
a2 = [bias sigmoid(z2)];

z3 = a2 * Theta2';
a3 = sigmoid(z3); # h(x), 5000 * 10 = 5000 * k
hx = a3;

J =  (-1/m) * sum(sum(Y .* log(hx) + ((1-Y) .* log(1-hx))));

# regularized cost function

s1 = input_layer_size;    # 400
s2 = hidden_layer_size;   # 25
s3 = k;                   # 10

t1 = Theta1(:, 2:end); # 25 * 400, remove thetas for bias
t2 = Theta2(:, 2:end); # 10 * 25

regTerm = (lambda / (2 * m)) * (sum(sum(t1 .^ 2)) + sum(sum(t2 .^ 2)));

J = J + regTerm;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients

d3 = a3 - Y; # layer3 errors, 5000 x 10
d2 = (d3 * Theta2)(:, 2:end) .* sigmoidGradient(z2); # layer 2 errors, 5000 * 25, no d2_0

D2 = (1/m) * (d3' * a2); # 10 * 26, layer2 partial derivative without the reg term
D1 = (1/m) * (d2' * a1); # 25 * 401, layer1 partial derivative without the reg term

Theta1_grad = D1;
Theta2_grad = D2;


% Part 3: Implement regularization with the cost function and gradients.

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
