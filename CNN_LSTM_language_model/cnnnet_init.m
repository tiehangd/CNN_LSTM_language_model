function cnnnet=cnnnet_init(opts)
%UNTITLED Summary of this function goes here

row = opts.emb_len + 1 - opts.cnn1.filterDim(1);
col = opts.phrase_len+1- opts.cnn1.filterDim(2);
cnnnet.paramsize = [opts.cnn1.filterDim, opts.cnn1.channel, opts.cnn1.numFilters];
cnnnet.W=1e-1*randn(cnnnet.paramsize);
cnnnet.b = zeros(opts.cnn1.numFilters, 1);
cnnnet.layersize=[row col opts.cnn1.numFilters];
cnnnet.velocity.W=zeros(size(cnnnet.W));
cnnnet.velocity.b=zeros(size(cnnnet.b));

end



