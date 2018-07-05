function [res] = input_bp( net,res,opts )
%NET_FF Summary of this function goes here
%   Detailed explanation goes here


    res.dzdy = tanh_ln(res.y,opts.dzdy);
    
    [res.dzdx, res.dzdw,res.dzdb] = backprop(res.x,net.Weight,res.dzdy);


end

