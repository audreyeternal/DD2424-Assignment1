function acc = ComputeAccuracy(X,y,W,b)
    num = 0;
    P =  EvaluateClassifier(X, W, b);
    for i = 1 : size(X,2)
        x = P(:,i);
        loc = find(x==max(x));
        if loc==y(i) 
            num = num +1;
        end
    end
    acc = num / size(X,2);
end

