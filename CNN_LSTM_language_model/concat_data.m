function [ opts ] = concat_data( opts )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%disp(size(data.train));
%disp(size(data.train_labels));
opts.train=opts.train(:,:,1:opts.phrase_len);
opts.train_labels=opts.train_labels(:,1:opts.phrase_len);
opts.test=opts.test(:,:,1:opts.phrase_len);
opts.test_labels=opts.test_labels(:,1:opts.phrase_len);

end

