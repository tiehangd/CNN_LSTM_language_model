function [res,Y] = soft_ff(In_Weight,In_Data)


    Y = In_Weight.Weight*In_Data.x+repmat(In_Weight.Bias,1,size(In_Data.x,2));
    E = exp(bsxfun(@minus, Y, max(Y,[],1))); %%%% why minus max(Y) here?
    L = sum(E,1);  %%  sums the first dimension, as the second dimension is the batch size;
    res = bsxfun(@rdivide, E, L);  %%  res and E should have the same dimension as Y;
    
end





