function [res] = tanh_ff(In_Data)

    res=tanh(In_Data.x);
    %res = 1 ./ (1 + exp(-In_Data.x));
    %opts.record_h=[opts.record_h,res];
        
    %disp('output h');
    %disp(res);
end