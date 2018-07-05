function [ data ] = rotate_data( data )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[dim1,dim2,dim3]=size(data.train);
rotate_data.train=zeros(dim1,dim3,dim2);

for i =1:dim2
    rotate_data.train(:,:,i)=data.train(:,i,:);

end


[dim1,dim2]=size(data.train_labels);
rotate_data.train_labels=zeros(dim2,dim1);

for i =1:dim1
    rotate_data.train_labels(:,i)=data.train_labels(i,:);

end



[dim1,dim2,dim3]=size(data.test);
rotate_data.test=zeros(dim1,dim3,dim2);

for i =1:dim2
    rotate_data.test(:,:,i)=data.test(:,i,:);

end


[dim1,dim2]=size(data.test_labels);
rotate_data.test_labels=zeros(dim2,dim1);

for i =1:dim1
    rotate_data.test_labels(:,i)=data.test_labels(i,:);

end


data.train_rot=rotate_data.train;
data.train_labels_rot=rotate_data.train_labels;
data.test_rot=rotate_data.test;
data.test_labels_rot=rotate_data.test_labels;

end

