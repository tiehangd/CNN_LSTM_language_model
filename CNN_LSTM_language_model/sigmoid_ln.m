function y = sigmoid_ln(x,dzdy)
    
    y = 1 ./ (1 + exp(-x));

    y = dzdy .* (y .* (1 - y));


end
