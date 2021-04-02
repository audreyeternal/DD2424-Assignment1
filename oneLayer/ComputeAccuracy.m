function acc = ComputeAccuracy(X,y,W,b)
    num = 0;
    P =  EvaluateClassifier(X, W, b)
    for i = 1 : size(X,2)
        loc = find(max(P(:,i)));
        if loc==y(i) 
            num = num +1;
        end
    end
    acc = num / size(X,2);
end

