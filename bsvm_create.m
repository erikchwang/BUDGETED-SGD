function model=bsvm_create(features,classes,option)

step_size=1/option.lambda;

for i=1:size(features,2)
    model.support_vectors(:,i)=features(:,i);
    model.coefficients(:,i)=classes(:,i)-circshift(classes(:,i),1);
end

model.coefficients=model.coefficients*step_size/size(features,2);

end