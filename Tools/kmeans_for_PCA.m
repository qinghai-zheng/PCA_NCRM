function [ACC_mean, ACC_std, NMI_mean, NMI_std] = kmeans_for_PCA(X,gt,k_repeat)
% The kmeans results with k repeats in the metric of ACC and NMI
% X is d*n here
% 
% Qinghai Zheng
% zhengqinghai@fzu.edu.cn

rng(1234,'twister');

ACC = zeros(k_repeat,1);
NMI = zeros(k_repeat,1);

data = X'; 
num_clu = length(unique(gt));

for i = 1:k_repeat
    [y_pre, ~] = litekmeans(data,num_clu,'MaxIter', 100,'Replicates',10);
    ACC(i) = Accuracy(y_pre,double(gt));
    NMI(i) = nmi(y_pre,gt);
end
  
ACC_mean = mean(ACC);
ACC_std = std(ACC);
NMI_mean = mean(NMI);
NMI_std = std(NMI);
















end