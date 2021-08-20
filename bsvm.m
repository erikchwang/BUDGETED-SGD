rng('shuffle');
run('load_data');

option.batch=100;
option.budget=1000;
option.epoch=30;
option.gamma=0.01;
option.lambda=0.00001;

model=bsvm_train(train_features,train_classes,option);
test_err=bsvm_err(test_features,test_classes,model,option);
fprintf('BSVM Accuracy: %.2f%%\n',(1-test_err)*100);

%batch=100,budget=1000,epoch=30,gamma=0.01,lambda=0.00001 => Accuracy=97.75%
%batch=100,budget=2000,epoch=30,gamma=0.01,lambda=0.00001 => Accuracy=97.89%
%batch=100,budget=4000,epoch=30,gamma=0.01,lambda=0.00001 => Accuracy=98.00%
%batch=100,budget=8000,epoch=30,gamma=0.01,lambda=0.00001 => Accuracy=98.09%
