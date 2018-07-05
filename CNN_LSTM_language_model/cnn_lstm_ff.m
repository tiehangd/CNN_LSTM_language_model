function [ net,cnnnet,res,opts ] = cnn_lstm_ff( net,cnnnet,preinputs,opts )
%NET_FF Summary of this function goes here
    


res.temp = cell(2, 1);
res.grad = cell(2, 1);
res.temp{1}.after = preinputs.data;
[res.temp{2}.after, res.temp{2}.linTrans] = cnnConvolve(res.temp{1}.after, cnnnet.W, cnnnet.b, opts.cnn1.nonLinearType);
res.temp{2}.sizeafter=size(res.temp{2}.after);
res.temp{2}.after_for_backprop=res.temp{2}.after;
res.temp{2}.after=permute(squeeze(res.temp{2}.after),[2,3,1]);
inputs.data=res.temp{2}.after;
inputs.labels=preinputs.labels(opts.win_len:end,:)';


    n_frames=opts.parameters.n_frames;
    
    n_cell_nodes=opts.parameters.n_cell_nodes;
    n_hidden_nodes=opts.parameters.n_hidden_nodes;
    batch_size=opts.parameters.batch_size;
    
    res.Cell{1}.x=zeros(n_cell_nodes,batch_size,'like',inputs.data);
    res.Hidden{1}.x=zeros(n_hidden_nodes,batch_size,'like',inputs.data);

    opts.err=zeros(1,n_frames,'like',inputs.data);
    if isfield(inputs,'labels')
        opts.err=zeros(2,n_frames,'like',inputs.data);
        opts.loss=zeros(1,n_frames,'like',inputs.data);
        %opts.err_comp=zeros(1,n_frames,'like',inputs.data);
    end
    
    opts.cost=0;
    extLabels=cell(n_frames,1);
    for f=1:n_frames
        
        extLabels{f} = zeros(opts.emb_len, batch_size);
        % disp(size(inputs.labels(:,f)));
        % disp(size(1 : batch_size));
        % disp(size(extLabels{f}));
        extLabels{f}(sub2ind(size(extLabels{f}), inputs.labels(:,f)', 1 : batch_size)) = 1;
        res.Gate{f}.x=[res.Hidden{f}.x;inputs.data(:,:,f)];    %%%% first dimension of input data is 67, second dimension is batch size, third dimension is time frame;
        res.Input{f}.x=res.Gate{f}.x;
        
        [res.Gate{f}.z,res.Gate{f}.y] = gate_ff(net.Gate,res.Gate{f});
        
        [res.Input{f}.z,res.Input{f}.y] = input_ff(net.Input,res.Input{f});
        
        res.Cell{f+1}.x=res.Gate{f}.z(1:n_cell_nodes,:).*res.Input{f}.z+res.Gate{f}.z(n_cell_nodes+1:2*n_cell_nodes,:).*res.Cell{f}.x;
        
        [res.Cell{f+1}.z] = tanh_ff(res.Cell{f+1});
        
        res.Hidden{f+1}.x=res.Gate{f}.z(2*n_cell_nodes+1:3*n_cell_nodes,:).*res.Cell{f+1}.z;
        
        res.Fit{f}.x=res.Hidden{f+1}.x;
        res.Fit{f}.class=inputs.labels(:,f);
        
        [res.Fit{f}.z,res.Fit{f}.y] = soft_ff(net.Softmax,res.Fit{f});

        opts.cost=opts.cost- mean(sum(extLabels{f} .* log(res.Fit{f}.z)));
        if isfield(inputs,'labels')
            opts.err(:,f)=error_multiclass(res.Fit{f}.class,res.Fit{f}.z);

        end

    
    end
    
    %opts.err=mean(opts.err.2)./opts.parameters.batch_size;
    opts.err_origin=mean(opts.err,2)./opts.parameters.batch_size;
    opts.loss_origin=opts.cost;
    
end

