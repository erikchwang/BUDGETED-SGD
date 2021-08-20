function model=bsvm_train(features,classes,option)

time_step=0;

for i=1:option.epoch
    temp_features=features;
    temp_classes=classes;
    
    while size(temp_features,2)>=option.batch
        select=randperm(size(temp_features,2),option.batch);
        select_features=temp_features(:,select);
        select_classes=temp_classes(:,select);
        temp_features(:,select)=[];
        temp_classes(:,select)=[];
        time_step=time_step+1;
        
        if time_step==1
            model=bsvm_create(select_features,select_classes,option);
        else
            model=bsvm_update(select_features,select_classes,time_step,model,option);
        end
        
        while size(model.support_vectors,2)>option.budget
            model=bsvm_merge(model,option);
        end
        
    end
    
    train_losses(i)=bsvm_loss(features,classes,model,option);
    train_errs(i)=bsvm_err(features,classes,model,option);
    fprintf('Epoch: %d -> Train Loss: %f, Train Err: %.2f%%\n',i,train_losses(i),train_errs(i)*100);
end

end