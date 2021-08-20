function model=bsvm_update(features,classes,time_step,current_model,option)

model.support_vectors=[];
model.coefficients=[];
step_size=1/(time_step*option.lambda);

for i=1:size(features,2)
    scores=current_model.coefficients*bsvm_kernel(current_model.support_vectors,features(:,i),option);
    scores=scores+1-classes(:,i);
    [max_score,max_index]=max(scores);
    
    if max_score~=scores'*classes(:,i)
        model.support_vectors=[model.support_vectors,features(:,i)];
        model.coefficients=[model.coefficients,classes(:,i)-circshift(eye(size(classes(:,i))),max_index-1)];
    end
end

model.support_vectors=[current_model.support_vectors,model.support_vectors];
model.coefficients=[current_model.coefficients*(1-step_size*option.lambda),model.coefficients*step_size/size(features,2)];

end