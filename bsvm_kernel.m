function products=bsvm_kernel(support_vectors,vector,option)

products=exp((support_vectors'*vector*2-sum(support_vectors.^2,1)'-sum(vector.^2))*option.gamma);

end