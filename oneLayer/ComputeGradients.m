function [grad_W, grad_b] = ComputeGradients(X, Y, P, W,~,lambda)
% �6�1 each column of X corresponds to an image and it has size d×n.
% �6�1 each column of Y (K×n) is the one-hot ground truth label for the corresponding column of X.
% �6�1 each column of P contains the probability for each label for the image
% in the corresponding column of X. P has size K×n.
% �6�1 grad W is the gradient matrix of the cost J relative to W and has size
% K×d.
% �6�1 grad b is the gradient vector of the cost J relative to b and has size
% K×1.
G = -(Y - P);
grad_W = 2*lambda*W + G*X'/size(X,2);
grad_b = sum(G,2)/size(X,2);
end

