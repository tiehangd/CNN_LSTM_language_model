function [res] = gate_bp(net,res,opts)
%NET_FF Summary of this function goes here

    res.dzdy = sigmoid_ln(res.y,opts.dzdy);

    [res.dzdx, res.dzdw, res.dzdb] = backprop(res.x,net.Weight,res.dzdy);
    

end


