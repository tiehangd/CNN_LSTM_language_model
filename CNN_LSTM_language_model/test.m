n_epoch=20;

%use_selective_sgd=0;

sgd_lr=1e-2;

opts.parameters.batch_size=100;
opts.parameters.n_hidden_nodes=30;
%opts.parameters.n_hidden_layer_nodes=100;
opts.parameters.n_cell_nodes=30;
opts.parameters.n_input_nodes=67;
opts.parameters.n_output_nodes=67;
opts.parameters.n_gates=3;
opts.parameters.n_frames=64;

opts.parameters.lr =sgd_lr;
opts.parameters.mom =0.9;
%opts.parameters.selective_sgd=use_selective_sgd;

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