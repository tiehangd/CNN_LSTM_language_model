function [opts]=test_lstm(net,opts)

    opts.training=0;


    opts.MiniBatchError=[];
    opts.MiniBatchLoss=[];
    
 
    
    for mini_b=1:opts.n_test_batch
        idx=1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size;
        
        
        if ~isfield(opts,'test_labels')
            inputs.data=opts.test(:,idx,:);
            output_data=inputs.data;
            inputs.data=inputs.data+randn(size(inputs.data))*0.005;
        else
             inputs.data=opts.test(:,idx,:);
             inputs.labels=opts.test_labels(idx,:);
            
        end
        %forward
        [ net,res,opts ] = lstm_ff( net,inputs,opts );

        %%%get your loss;
        if ~isfield(opts,'test_labels')
        
            loss=0;
            for f=1:opts.parameters.n_frames
               opts.lstm_dzdy{f}=res.Fit{f}(end).x -output_data(:,:,f); 
               loss=loss+sum(opts.lstm_dzdy{f}(:).^2)./opts.parameters.batch_size;
            end
        else
           %%%%%fill here 
            %disp([' Minibatch error: ', num2str(opts.err(1)), ' Minibatch loss: ', num2str(opts.err(2))])    
        end

        opts.MiniBatchError=[opts.MiniBatchError;gather(opts.err(1))];
        opts.MiniBatchLoss=[opts.MiniBatchLoss;gather(opts.loss)]; 
      
    end
    
    opts.results.TestEpochError=[opts.results.TestEpochError;mean(opts.MiniBatchError(:))];
    opts.results.TestEpochLoss=[opts.results.TestEpochLoss;mean(opts.MiniBatchLoss(:))];
      
end


