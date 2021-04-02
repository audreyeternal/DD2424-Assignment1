clc
clear
close all

lambda = 0;
[trainX,labelY,y] = LoadBatch('data_batch_1.mat');
mean_X = mean(trainX, 2);
std_X = std(trainX, 0, 2);
trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]);
trainX = trainX ./ repmat(std_X, [1, size(trainX, 2)]);

%initialize parameter W and b:
K = size(labelY,1);
d = size(trainX,1);
W = 0.1*randn(K,d);
b = 0.1*randn(K,1);
n_batch = 2;
eta = .01;
n_epochs = 5;
GDparams.n_batch = n_batch;
GDparams.eta = eta;
GDparams.n_epochs = n_epochs;

%P = EvaluateClassifier(trainX, W, b);
%%
%compare the result between the numerical methods and the analytical
%methods.
% [ngrad_b_slow, ngrad_W_slow] = ComputeGradsNumSlow(trainX(1:20,1), labelY(:,1),W(:,1:20), b, lambda, 1e-6);
% [ngrad_b, ngrad_W] = ComputeGradsNum(trainX(1:20,1), labelY(:,1),W(:,1:20), b, lambda, 1e-6);
% [agrad_W, agrad_b] = ComputeGradients(trainX(1:20,1),labelY(:,1), P(:,1), W(:,1:20),b,lambda);
%%
[Wstar, bstar] = miniBatchGD(trainX, labelY, y, GDparams, W, b, lambda);







