function res = soft_bp(net,res,opts)
%NET_FF Summary of this function goes here

for f=1:opts.parameters.n_frames

    res.Fit{f}.dzdy = softloglossbp(res.Fit{f}.y,res.Fit{f}.class);  %% this propagates to output of hidden unit;

    [res.Fit{f}.dzdx, res.Fit{f}.dzdw,res.Fit{f}.dzdb] = backprop(res.Fit{f}.x,net.Softmax.Weight,res.Fit{f}.dzdy);
    
end
    
    
end

