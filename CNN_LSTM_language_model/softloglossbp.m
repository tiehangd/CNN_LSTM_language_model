function Y = softloglossbp(X,c)

dim_out=1;

n_class=size(X,dim_out);

Xmax = max(X,[],dim_out);
ex = exp(bsxfun(@minus, X, Xmax));

idx=c(:)'+n_class*[0:size(X,dim_out+1)-1];

Y = bsxfun(@rdivide, ex, sum(ex,dim_out));
Y(idx)=Y(idx)-1;    %% the positions that labels appear;

  
end


