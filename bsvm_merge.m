function model=bsvm_merge(current_model,option)

model.support_vectors=current_model.support_vectors;
model.coefficients=current_model.coefficients;
[~,first_index]=min(sum(current_model.coefficients.^2,1));
products=bsvm_kernel(current_model.support_vectors,current_model.support_vectors(:,first_index),option);
basises=current_model.coefficients+current_model.coefficients(:,first_index)+eps;
alphas=sum(current_model.coefficients(:,first_index)./basises,1);
betas=sum(current_model.coefficients./basises,1);

for i=1:size(current_model.support_vectors,2)
    if i==first_index
        degradations(i)=Inf;
    else
        partition=bsvm_merge_search(-10,10,0.00001,products(i),alphas(i),betas(i));
        support_vectors(:,i)=current_model.support_vectors(:,first_index)*partition+current_model.support_vectors(:,i)*(1-partition);
        coefficients(:,i)=current_model.coefficients(:,first_index)*(products(i)^((1-partition)^2))+current_model.coefficients(:,i)*(products(i)^(partition^2));
        degradations(i)=sum(current_model.coefficients(:,first_index).^2)+sum(current_model.coefficients(:,i).^2)+2*products(i)*current_model.coefficients(:,first_index)'*current_model.coefficients(:,i)-sum(coefficients(:,i).^2);
    end
    
end

[~,second_index]=min(degradations);
model.support_vectors(:,[first_index,second_index])=[];
model.coefficients(:,[first_index,second_index])=[];
model.support_vectors=[model.support_vectors,support_vectors(:,second_index)];
model.coefficients=[model.coefficients,coefficients(:,second_index)];

end

function partition=bsvm_merge_search(left_bound,right_bound,tolerance,product,alpha,beta)

golden_ratio=(sqrt(5)+1)/2;
left_bound_plus=right_bound-(right_bound-left_bound)/golden_ratio;
right_bound_minus=left_bound+(right_bound-left_bound)/golden_ratio;

while abs(left_bound_plus-right_bound_minus)>tolerance
    left_bound_plus_value=-(alpha*product^((1-left_bound_plus)^2)+beta*product^(left_bound_plus^2));
    right_bound_minus_value=-(alpha*product^((1-right_bound_minus)^2)+beta*product^(right_bound_minus^2));
    
    if left_bound_plus_value<right_bound_minus_value
        right_bound=right_bound_minus;
    else
        left_bound=left_bound_plus;
    end
    
    left_bound_plus=right_bound-(right_bound-left_bound)/golden_ratio;
    right_bound_minus=left_bound+(right_bound-left_bound)/golden_ratio;
end

partition=(left_bound+right_bound)/2;

end