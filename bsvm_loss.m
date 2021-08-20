function loss=bsvm_loss(features,classes,model,option)

loss=0;

for i=1:size(features,2)
    scores=model.coefficients*bsvm_kernel(model.support_vectors,features(:,i),option);
    scores=scores+1-classes(:,i);
    loss=loss+max(scores)-scores'*classes(:,i);
end

loss=loss/size(features,2);

end