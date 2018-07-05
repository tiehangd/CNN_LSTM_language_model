function err = error_multiclass(labels, res)

predictions=res;

[~,predictions] = sort(predictions, 1, 'descend'); %% for each data in the mini batch, we sort it to be highest to lowest, prediction is the index of the highest;    
save('pred.mat','predictions');
if numel(labels) == size(predictions, 2)
  labels = reshape(labels,1,[]);
end


error = ~bsxfun(@eq, predictions, labels);
err(1,1) = sum(error(1,:));
%disp(size(error,1));
if size(error,1)>=5
    err(2,1) = sum(min(error(1:5,:),[],1));
else
    err(2,1)=sum(min(error(1:end,:),[],1));
end



