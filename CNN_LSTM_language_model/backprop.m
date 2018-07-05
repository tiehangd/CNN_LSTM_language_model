function [ y, dzdw,dzdb ] = backprop( I,weight,dzdy )
%FAST_MLP_LAYER Summary of this function goes here
    
    y=weight'*dzdy;
    
    dzdb=mean(dzdy,2);  %minibatch averaging    

    dzdy=permute(dzdy,[1,3,2]);  %% let batch size to be the last dimension;
    I=permute(I,[3,1,2]);
    dzdw=bsxfun(@times,dzdy,I);
    dzdw=mean(dzdw,3);
    
end




