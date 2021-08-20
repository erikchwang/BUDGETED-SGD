function err=bsvm_err(features,classes,model,option)

err=0;

for i=1:size(features,2)
    scores=model.coefficients*bsvm_kernel(model.support_vectors,features(:,i),option);
    err=err+(max(scores)~=scores'*classes(:,i));
end

err=err/size(features,2);

end