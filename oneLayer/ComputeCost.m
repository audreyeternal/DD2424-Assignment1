function J = ComputeCost(X,Y,W,b,lambda)
%each column of X corresponds to an image and X has size d*n.
%each column of Y (K¡Án) is the one-hot ground truth label for the corresponding column of X or Y is the (1*n) vector of ground truth labels.
%J is a scalar corresponding to the sum of the loss of the network's
% predictions for the images in X relative to the ground truth labels and
% the regularization term on W.
P = EvaluateClassifier(X,W,b);
loss = zeros(1,size(P,2));
for i=1:size(loss,2)
loss(1,i) = -log10(Y(:,i)'*P(:,i));
end
J = mean(loss);%the cross-entropy loss term
J = J + lambda*sum(W.*W,'all'); %plus regularization term
end

