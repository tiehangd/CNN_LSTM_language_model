function [res,Y] = gate_ff(Gate,Input)

    %disp(size(Gate.Weight));
    %disp(size(Input.x));
    Y=Gate.Weight*Input.x+repmat(Gate.Bias,1,size(Input.x,2));
    res = 1 ./ (1 + exp(-Y));

end

