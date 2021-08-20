train_features=load_images('train_images');
train_classes=load_labels('train_labels');
test_features=load_images('test_images');
test_classes=load_labels('test_labels');

function features=load_images(filename)

file=fopen(filename);
assert(file~=-1,['Could not open',' ',filename]);
magic=fread(file,1,'int32',0,'ieee-be');
assert(magic==2051,['Bad magic number in',' ',filename]);
count=fread(file,1,'int32',0,'ieee-be');
rows_num=fread(file,1,'int32',0,'ieee-be');
cols_num=fread(file,1,'int32',0,'ieee-be');
features=fread(file,Inf,'unsigned char');
features=reshape(features,rows_num*cols_num,count);
features=double(features)/255;
fclose(file);

end

function classes=load_labels(filename)

file=fopen(filename);
assert(file~=-1,['Could not open',' ',filename]);
magic=fread(file,1,'int32',0,'ieee-be');
assert(magic==2049,['Bad magic number in',' ',filename]);
count=fread(file,1,'int32',0,'ieee-be');
classes=fread(file,Inf,'unsigned char');
assert(size(classes,1)==count,'Mismatch in label count');
classes=bsxfun(@eq,unique(classes),classes');
fclose(file);

end