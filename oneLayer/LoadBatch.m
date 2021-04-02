function [X,Y,y] = LoadBatch(filename)
% • X contains the image pixel data, has size d×n, is of type double or
% single and has entries between 0 and 1. n is the number of images
% (10000) and d the dimensionality of each image (3072=32×32×3).
% • Y is K×n (K= # of labels = 10) and contains the one-hot representation
% of the label for each image.
% • y is a vector of length n containing the label for each image. A note
% of caution. CIFAR-10 encodes the labels as integers between 0-9 but
% Matlab indexes matrices and vectors starting at 1. Therefore it may be
% easier to encode the labels between 1-10.
addpath Datasets/cifar-10-batches-mat/;
A = load(filename);
X = A.data';
X = double(X);
Y = zeros(10,10000);
for i = 1:10000
    Y(round(A.labels(i)+1),i)=1;
end
y = A.labels+1;
y = y';







end