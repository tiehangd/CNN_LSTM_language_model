
n_epoch=2;

%use_selective_sgd=0;

sgd_lr=1e-2;
opts.win_len=8;

opts.parameters.batch_size=100;
opts.parameters.n_hidden_nodes=30;  %% this lies a difference;
%opts.parameters.n_hidden_layer_nodes=100;
opts.parameters.n_cell_nodes=30;
opts.emb_len=67;

opts.parameters.n_gates=3;
opts.phrase_len=24;
opts.parameters.n_frames=opts.phrase_len-opts.win_len+1;    %% modified to be 24;

opts.parameters.lr =sgd_lr;
opts.parameters.mom =0.5;
opts.parameters.mom2 =0.95;
%opts.parameters.selective_sgd=use_selective_sgd;
opts.it=0; %% to indicate the increase of momentum;
opts.momIncrease = 20;
opts.n_epoch=n_epoch;
opts.results=[];
opts.results.TrainEpochError=[];
opts.results.TestEpochError=[];
opts.results.TrainEpochLoss=[];
opts.results.TestEpochLoss=[];
%opts.RecordStats=1;
opts.results.TrainLoss=[];
opts.results.TrainError=[];

opts=PrepareData_Char_LSTM(opts);
[ opts ] = concat_data( opts ); %% newly added;

opts.n_batch=floor(opts.n_train/opts.parameters.batch_size);
opts.n_test_batch=floor(opts.n_test/opts.parameters.batch_size);

opts.parameters.current_ep=1;
start_ep=opts.parameters.current_ep;


opts.num_filter1=37;
opts.cnn1.alpha = 1e-1;
opts.cnn1.momentum = 0.8;
opts.cnn1.channel=1;
opts.cnn1.filterDim = [opts.emb_len opts.win_len];
opts.cnn1.numFilters = opts.num_filter1;
opts.cnn1.nonLinearType = 'sigmoid';
opts.input.dimension=[opts.emb_len,opts.phrase_len,1];


opts.parameters.n_input_nodes=opts.cnn1.numFilters;
opts.parameters.n_output_nodes=opts.emb_len;
cnnnet=cnnnet_init(opts);
net=net_init_char_lstm(opts);
opts=rotate_data(opts);

d = opts.input.dimension;
opts.train_rot = reshape(opts.train_rot,d(1),d(2),d(3),[]);


opts.record_h=[];
opts.record_g=[];
opts.store.error=[];
opts.store.loss=[];

for ep=start_ep:opts.n_epoch
    
    [net,cnnnet, opts,res]=train_cnn_lstm(net,cnnnet,opts); 

    disp('ep');
    disp(ep);
    
    
end




