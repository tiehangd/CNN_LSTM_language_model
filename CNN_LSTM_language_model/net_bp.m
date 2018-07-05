function [ net,res,opts ] = net_bp( net,res,opts )
%NET_FF Summary of this function goes here
%   Detailed explanation goes here

    if opts.use_gpu
        opts.dzdy=gpuArray(single(opts.dzdy));
    end    
    res(numel(net.layers)+1).dzdx = opts.dzdy ;

    for layer=numel(net.layers):-1:1
            
        opts.current_layer=layer;
        switch net.layers{layer}.type

            case 'conv'
                if isfield(net.layers{1,layer},'stride')
                    if(length(net.layers{1,layer}.stride)==1)
                        net.layers{1,layer}.stride=ones(1,2)*net.layers{1,layer}.stride;
                    end
                else
                   net.layers{1,layer}.stride=1;
                end
                
                
                if isfield(net.layers{1,layer},'pad')
                    if(length(net.layers{1,layer}.pad)==1)
                        net.layers{1,layer}.pad=ones(1,4)*net.layers{1,layer}.pad;
                    end
                else
                   net.layers{1,layer}.pad=[];
                end
                
                [res(layer).dzdx, res(layer).dzdw,res(layer).dzdb,opts] = fast_conv_layer( res(layer).x,net.layers{1,layer}.weights{1},net.layers{1,layer}.weights{2},net.layers{1,layer}.stride,net.layers{1,layer}.pad,res(layer+1).dzdx,opts );
                
            case 'mlp'                             
                [res(layer).dzdx, res(layer).dzdw,res(layer).dzdb] = fast_mlp_layer( res(layer).x,net.layers{1,layer}.weights{1},net.layers{1,layer}.weights{2},res(layer+1).dzdx );
            
            case 'bnorm'
                [~,res(layer).dzdx,res(layer).dzdw,res(layer).dzdb] = bnorm( net,res(layer).x,layer,res(layer+1).dzdx ,opts );
            case {'normalize', 'lrn'}
                res(layer).dzdx = lrn(res(layer).x, net.layers{1,layer}.param(1),net.layers{1,layer}.param(2),net.layers{1,layer}.param(3),net.layers{1,layer}.param(4),res(layer+1).dzdx ) ;
                        
            case 'relu'
                res(layer).dzdx = relu(res(layer).x, res(layer+1).dzdx) ;
            case 'leaky_relu'
                res(layer).dzdx = leaky_relu(res(layer).x, res(layer+1).dzdx) ;   
            case 'sigmoid'
                res(layer).dzdx = sigmoid_ln(res(layer).x,res(layer+1).dzdx );
            case 'tanh'
                res(layer).dzdx = tanh_ln(res(layer).x,res(layer+1).dzdx );
            case 'pad'
                [res(layer).x,res(layer).dzdx]=pad_data(res(layer+1).x,net.layers{1,layer}.pad,res(layer+1).dzdx);
            case 'pool'
                
                if isfield(net.layers{1,layer},'pad')
                    if(length(net.layers{1,layer}.pad)==1)
                        net.layers{1,layer}.pad=ones(1,4)*net.layers{1,layer}.pad;
                    end
                else
                   net.layers{1,layer}.pad=[];
                end
                                
                if isfield(net.layers{1,layer},'stride')
                    if(length(net.layers{1,layer}.stride)==1)
                        net.layers{1,layer}.stride=ones(1,2)*net.layers{1,layer}.stride;
                    end
                end
                
                res(layer).dzdx = maxpool(res(layer).x, net.layers{1,layer}.pool, net.layers{1,layer}.stride,net.layers{1,layer}.pad,res(layer+1).dzdx,res(layer+1).from);

            case 'softmaxloss'
                res(layer).dzdx = softmaxlogloss(res(layer).x, res(1).class, res(layer+1).dzdx) ;


        end

    end

end

