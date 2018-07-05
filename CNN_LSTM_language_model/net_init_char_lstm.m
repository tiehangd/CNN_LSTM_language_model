function [net] = net_init_char_lstm(opts)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST

rng('default');
rng(0);

f=1/100;

n_hidden_nodes=opts.parameters.n_hidden_nodes;
n_input_nodes=opts.parameters.n_input_nodes;
n_output_nodes=opts.parameters.n_output_nodes;
n_cell_nodes=opts.parameters.n_cell_nodes;
n_gates=opts.parameters.n_gates;


net.Gate.Weight=f*randn(n_gates*n_cell_nodes,n_hidden_nodes+n_input_nodes);
net.Gate.Bias=zeros(n_gates*n_cell_nodes,1);
net.Input.Weight=f*randn(n_cell_nodes,n_hidden_nodes+n_input_nodes);
net.Input.Bias=zeros(n_cell_nodes,1);

net.Softmax.Weight=f*randn(n_output_nodes,n_hidden_nodes);
net.Softmax.Bias=zeros(n_output_nodes,1,'single');


end




