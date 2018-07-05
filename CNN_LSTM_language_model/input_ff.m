function [res,Y] = input_ff(In_Weight,In_Data)

    Y=In_Weight.Weight*In_Data.x+repmat(In_Weight.Bias,1,size(In_Data.x,2));
    res = tanh(Y);
    %opts.record_g=[opts.record_g,res];
    %disp('input g');
    %disp(res);
    
end



