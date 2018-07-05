function Y = softmaxlogloss(X,c,dzdy)

if(length(size(X))>2)
    sz = size(X);
    if(sz(1)>1||sz(2)>1) error('Size error in softmax log loss.'); end
    dim_out=3;
end
if(length(size(X))<=2)
    dim_out=1;
end
n_class=size(X,dim_out);

Xmax = max(X,[],dim_out) ;
ex = exp(bsxfun(@minus, X, Xmax)) ;

idx=c(:)'+n_class*[0:size(X,dim_out+1)-1];   
if nargin <= 2
    %forward
    Y = log(sum(ex,dim_out)) +Xmax -reshape(X(idx),size(Xmax));
    %Y = sum(Y,dim_out+1);% sum of batch loss
else
    %bp
    Y = bsxfun(@rdivide, ex, sum(ex,dim_out));
    Y(idx)=Y(idx)-1;
    Y = bsxfun(@times, Y, dzdy);
end


