function P =  EvaluateClassifier(X, W, b)
% • each column of X corresponds to an image and it has size d×n.
% • W and b are the parameters of the network.
% • each column of P contains the probability for each label for the image
% in the corresponding column of X. P has size K×n.
P = W*X + b;
P = exp(P)./ sum(exp(P));
end

