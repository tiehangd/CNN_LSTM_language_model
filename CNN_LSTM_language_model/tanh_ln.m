function y = tanh_ln(x,dzdy)


    y = dzdy.*(4./(exp(x)+exp(-x)).^2);


end

