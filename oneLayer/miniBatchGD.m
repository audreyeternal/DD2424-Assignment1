function [Wstar, bstar] = miniBatchGD(X, Y, y, GDparams, W, b, lambda)
% where:
% X contains all the training images, 
% Y the labels for the training images, 
% W, b are the initial values for the network¡¯s parameters,
% lambda is the regularization factor in the cost function
% GDparams is an object containing the parameter values n batch, eta
% and n epochs.
n_batch = GDparams.n_batch;
eta = GDparams.eta ; %learning rate;
n_epochs = GDparams.n_epochs ; 
iter_b = b;
iter_W = W;
acc_list = zeros(1,n_epochs);

for i = 1:n_epochs    
    loc = randperm(size(X,2));
    X = X(:,loc);
    Y = Y(:,loc);
    y = y(loc);
    for j=1:size(X,2)/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);
        %[grad_b, grad_W] = ComputeGradsNum(Xbatch, Ybatch,iter_W, iter_b, lambda, 1e-6);
        P = EvaluateClassifier(Xbatch, iter_W, iter_b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch,P,iter_W, iter_b, lambda);
        iter_b = iter_b - eta*grad_b;
        iter_W = iter_W - eta*grad_W;
    end
    acc = ComputeAccuracy(X,y,iter_W,iter_b);
    acc_list(i) = acc;
    %print("---------epoch"+num2str(i)+":"+"accuracy = "+num2str(acc)+"----------");
    %print("1")
end
figure 
plot(1:n_epochs,acc_list);
Wstar = iter_W;
bstar = iter_b;
end

