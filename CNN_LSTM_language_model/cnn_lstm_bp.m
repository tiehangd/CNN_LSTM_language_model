function [net,cnnnet,res,opts] = cnn_lstm_bp(net,cnnnet,res,opts)
%SLTM_BP Summary of this function goes here

    n_frames=opts.parameters.n_frames;
    n_cell_nodes=opts.parameters.n_cell_nodes;
    n_hidden_nodes=opts.parameters.n_hidden_nodes;
    batch_size=opts.parameters.batch_size;
    
    %1: calculate the gradients of the data fitting transform

    res=soft_bp(net,res,opts);
    
    
    %2: BPTT: calculate the gradient wrt memory cell 

    dzdct=0;%accumulated gradient in later time frames
    res.Gate{n_frames+1}.dzdx=zeros(size(res.Gate{n_frames}.x));
    res.Input{n_frames+1}.dzdx=zeros(size(res.Gate{n_frames}.x));
    for f=n_frames:-1:1
        %%the gradient here is the previous accumulated gradient + the one from the output gate 
        
        opts.dzdy=res.Gate{f}.z(2*n_cell_nodes+1:3*n_cell_nodes,:).*(res.Fit{f}.dzdx+res.Gate{f+1}.dzdx(1:size(res.Hidden{f}.x,1),:)+res.Input{f+1}.dzdx(1:size(res.Hidden{f}.x,1),:));
        res.Cell{f+1} = hidden_bp(res.Cell{f+1},opts);
        res.Cell{f+1}.dzdx=res.Cell{f+1}.dzdx+dzdct;
        %%bp to previous time frame
        dzdct=res.Cell{f+1}.dzdx.*res.Gate{f}.z(n_cell_nodes+1:2*n_cell_nodes,:);
        
        opts.dzdy= res.Gate{f}.z(1:n_cell_nodes,:).*res.Cell{f+1}.dzdx;
        [res.Input{f}] = input_bp(net.Input,res.Input{f},opts);
        
        res.Hidden{f+1}.dzdx=res.Fit{f}.dzdx+res.Gate{f+1}.dzdx(1:size(res.Hidden{f}.x,1),:)+res.Input{f+1}.dzdx(1:size(res.Hidden{f}.x,1),:);
        opts.dzdy=[res.Input{f}.z.*res.Cell{f+1}.dzdx;...
        res.Cell{f}.x .*res.Cell{f+1}.dzdx;...
        res.Hidden{f+1}.dzdx.*res.Cell{f+1}.z];
        res.Gate{f} = gate_bp(net.Gate,res.Gate{f},opts);
        
    end
    
    %res.Cell{1}(end+1).x=0;  %just some padding
    res.Cell{1}.dzdx=0;
    

    %4: calculate the gradients of the input,forget,output gates:

    
    
    %%%accumulate gradients in all time frames
    
    [res.Fit_all.ac_dzdw,res.Fit_all.ac_dzdb]=average_gradients(res.Fit,opts);
    [res.Input_all.ac_dzdw,res.Input_all.ac_dzdb]=average_gradients(res.Input,opts);
    [res.Gate_all.ac_dzdw,res.Gate_all.ac_dzdb]=average_gradients(res.Gate,opts);
    %[res.Cell.ac_dzdw,res.Cell.ac_dzdb]=average_gradients(res.Cell);
    
    

    
    res.temp{2}.gradtemp=zeros(res.temp{2}.sizeafter);
    
    seq_len=opts.parameters.n_frames;
    for ii=1:seq_len
        res.Input{ii}.dzdx_x=res.Input{ii}.dzdx((opts.parameters.n_hidden_nodes+1):end,:)+res.Gate{ii}.dzdx((opts.parameters.n_hidden_nodes+1):end,:);
        %disp(size(res.Input{ii}.dzdx_x));
        %disp(size(res.temp{2}.gradtemp(1,ii,:,:)));
        res.temp{2}.gradtemp(1,ii,:,:)=res.Input{ii}.dzdx_x;
    
    end
    
    switch opts.cnn1.nonLinearType
        case 'sigmoid'

            res.temp{2}.gradBefore=res.temp{2}.gradtemp.*res.temp{2}.after_for_backprop.*(1 - res.temp{2}.after_for_backprop);
        
    end
    
    tempW = zeros([size(cnnnet.W) batch_size]);
    numInputMap = size(tempW, 3);
    numOutputMap = size(tempW, 4);
    for i = 1 : batch_size
        for nI = 1 : numInputMap
            for nO = 1 : numOutputMap
                
                tempW(:,:,nI,nO,i) = conv2(res.temp{1}.after(:,:,nI,i), rot90(res.temp{2}.gradBefore(:,:,nO,i), 2), 'valid');   

            end
        end
    end
    
    res.cnngrad{2}.W = mean(tempW,5);
    tempb = mean(sum(sum(res.temp{2}.gradBefore)),4);
    res.cnngrad{2}.b = tempb(:);
    %disp('size(tempb)');
    %disp(size(tempb));
end


