function W= fun_Laplace_PCA(X,dim_tar,lambda,gamma)
% fun_Laplace_PCA
% The Laplace function is leveraged in the PCA learning. 
% Laplace: f(x) = lambda*(1-exp(-1*x/gamma))
%          g(x) = x^2
%          f(x) = lambda*(1-exp(-1*(g(x))^0.5/gamma))
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
    D_tmp_i_sqrt = D_tmp_i.^(0.5);
    D_tmp_i_sqrt_neg = D_tmp_i.^(-0.5);
    d = (0.5*lambda/gamma)*exp(-1*D_tmp_i_sqrt/gamma).*D_tmp_i_sqrt_neg;
    
    D = diag(d);
    [W,~,~] = svds(X*((D)^(0.5)),dim_tar);
    iter_cur = iter_cur +1;
end


