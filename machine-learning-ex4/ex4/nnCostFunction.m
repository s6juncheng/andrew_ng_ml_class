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

X = [ones(m, 1) X];

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
% Forward propgation prediction with parater THETA (nn_params)
z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
% Compute for output layer (predictions)
predictions_all = sigmoid(z3);  % 5000 X 10 dimention

% Rewrite the training labels y into a matrix format Y 5000 X 10 dimention
% e.g. if y(1)=1, then Y(1,2)=1, the others of Y(1:)=0
Y = zeros(m, num_labels);
for i = 1:num_labels
    Y(:,i) = y==i;
end

Theta1_sqr = Theta1.^2;
Theta2_sqr = Theta2.^2;
J = (1/m) * sum(sum( -Y.*log(predictions_all) - (1-Y).*log(1-predictions_all) )) + ...
    (lambda/(2*m)) * ( sum(sum(Theta1_sqr(:,2:(input_layer_size+1)))) + sum(sum(Theta2_sqr(:,2:(hidden_layer_size+1)))) );

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
d3 = predictions_all - Y;  % 5000 X 10
%fprintf('d3: %f \n', size(d3));
%fprintf('%f ',size(d3 * Theta2(:,2:end)));
%fprintf('%f ',  size(Theta2(:,2:end)));
%fprintf('%f \n', size( sigmoidGradient(z2(:,2:end))))
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);
%fprintf('d2: %f \n', size(d2));
% (5000 X 10) * 10 X 25 = 5000 X 25   
Delta1 = d2' * X;
%fprintf('Delta1: %f \n', size(Delta1));
Theta1_grad = (1/m) .* Delta1;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Delta2 = d3' * a2;
%fprintf('Delta2: %f \n', size(Delta2));
Theta2_grad = (1/m) .* Delta2;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
