function [net,cnnnet,opts, res]=train_cnn_lstm(net,cnnnet,opts)

    opts.training=1;

    opts.MiniBatchError=[];
    opts.MiniBatchLoss=[];


    tic
    
    opts.order=randperm(opts.n_train);
    
    batch_size=opts.parameters.batch_size;
    
    for mini_b=1:opts.n_batch
        
        opts.it=opts.it+1;
        
        if opts.it == opts.momIncrease  %% first 20 minibatch uses momentum 0.5, then momentum is increased to 0.9;
            opts.parameters.mom = opts.parameters.mom2;

        end;
        
        idx=opts.order(1+(mini_b-1)*batch_size:mini_b*batch_size);

        
        inputs.data=opts.train_rot(:,:,:,idx);
        inputs.labels=opts.train_labels_rot(:,idx);
        
        [net,cnnnet,res,opts] = cnn_lstm_ff(net,cnnnet,inputs,opts);

        [net,cnnnet,res,opts] = cnn_lstm_bp(net,cnnnet,res,opts);
        
        
        

        %inputs.data=opts.train(:,idx,:);
        %inputs.labels=opts.train_labels(idx,:);
        
        %forward
        
        
        disp([' Minibatch error: ', num2str(opts.err_origin(2)), ' Minibatch loss: ', num2str(opts.loss_origin)])
        
        
        opts.store.error=[opts.store.error,opts.err_origin(2)];
        opts.store.loss=[opts.store.loss,opts.loss_origin];
        
        opts.MiniBatchError=[opts.MiniBatchError;gather( opts.err(1))];
        opts.MiniBatchLoss=[opts.MiniBatchLoss;gather( opts.loss)];
        
        [  net.Gate,opts ] = adam(net.Gate,res.Gate_all,opts);
        
        [  net.Input,opts ] = adam(net.Input,res.Input_all,opts);
        
        %[  net.Cell.Weight,net.Cell.Bias,opts ] = adam(net.Cell,res.Cell,opts);  
        
        [  net.Softmax,opts ] = adam(net.Softmax,res.Fit_all,opts);
        
         cnnnet.velocity.W = opts.parameters.mom * cnnnet.velocity.W + opts.cnn1.alpha * res.cnngrad{2}.W;
         cnnnet.W = cnnnet.W - cnnnet.velocity.W;
         cnnnet.velocity.b = opts.parameters.mom * cnnnet.velocity.b + opts.cnn1.alpha * res.cnngrad{2}.b;
         cnnnet.b = cnnnet.b - cnnnet.velocity.b;
        
        
        

    end
    
    opts.results.TrainEpochError=[opts.results.TrainEpochError;mean(opts.MiniBatchError(:))];
    opts.results.TrainEpochLoss=[opts.results.TrainEpochLoss;mean(opts.MiniBatchLoss(:))];
    
    toc;

end


