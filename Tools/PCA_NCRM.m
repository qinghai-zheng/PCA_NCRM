function W= PCA_NCRM(X,opts)
% PCA_NCRM
% Non-decreasing Concave Regularized Minimization for Principal Component Analysis 
% We considers Laplace, ETP, and Geman here
% 
% X: d*n
% 
% zhengqinghai@fzu.edu.cn
% 2024/07/03

dim_tar = 50;
if isfield(opts,'dim_tar')
    dim_tar = opts.dim_tar;
end

mode = 'Laplace';
if isfield(opts,'mode')
    mode = opts.mode;
end

p = 0.5;
if isfield(opts,'p')
    p = opts.p;
end

lambda = 1;
if isfield(opts,'lambda')
    lambda = opts.lambda;
end

gamma = 1;
if isfield(opts,'gamma')
    gamma = opts.gamma;
end

if strcmp(mode, 'ETP')
    W = fun_ETP_PCA(X, dim_tar, lambda, gamma);
elseif strcmp(mode, 'Geman')
    W = fun_Geman_PCA(X, dim_tar, lambda, gamma);
elseif strcmp(mode, 'Laplace')
    W = fun_Laplace_PCA(X, dim_tar, lambda, gamma);
else
    error('mode does not support!');
end


