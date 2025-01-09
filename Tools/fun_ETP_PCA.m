function W= fun_ETP_PCA(X,dim_tar,lambda, gamma)
% ETP_PCA
% The ETP function is leveraged in the PCA learning. 
% ETP: f(x) = (lambda/(1-exp(-1*gamma)))*(1-exp(-1*gamma*x))
%      g(x) = x^2
%      f(x) = f(g(x)) = (lambda/(1-exp(-1*gamma)))*(1-exp(-1*gamma*(g(x))^0.5))
%
% zhengqinghai@fzu.edu.cn
% 2024/07/02

[dim_ori,~] = size(X);
W = orth(rand(dim_ori,dim_tar));

iter_cur = 1;
iter_max = 20;

while  iter_cur <= iter_max

    D_tmp = X - W*W'*X;
    D_tmp_i = (sum(D_tmp.*D_tmp,1))'; 
    d = (0.5*lambda*gamma/(1-exp(-1*gamma)))*exp(-1*gamma*D_tmp_i.^0.5).*((D_tmp_i).^(-0.5));
    
    D = diag(d);    
    [W,~,~] = svds(X*((D).^(0.5)),dim_tar);
    iter_cur = iter_cur +1;
end


