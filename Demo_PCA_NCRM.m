clear,clc;
addpath('Dataset_Used','Metrics','Tools');

file_list = dir('Dataset_Used');
dataset_num = length(file_list) - 2;
dataset_list = cell(dataset_num, 1);
for i = 1 : dataset_num
    dataset_list{i} = file_list(i+2).name;
end
k_repeat = 30;

opts.lambda = 1;

gamma_list = [0.001, 0.01, 0.1, 1, 10, 100];
% for Laplace, we fix gamma = 0.001

for dataset_idx = 2: dataset_num
    fprintf('Running on the following dataset: %s \n',cell2mat(dataset_list(dataset_idx)));
    for gamma_idx = 1:length(gamma_list)
        gamma = gamma_list(gamma_idx);
        load(cell2mat(dataset_list(dataset_idx)));
        gt = minmax_scaling(:,1)+1;
        X_tmp = minmax_scaling(:,2:end);
        X = X_tmp';
        c = length(unique(gt));
        [dim_ori, sample_num] = size(X);
        dim_dir = c-1;
        opts.dim_tar = dim_dir;
        if dim_ori <= dim_dir
            dim_dir = ceil(dim_ori/2);
        end
        
        fprintf('Target Dimension: %d, gamma %d \n', dim_dir, gamma);
        opts.gamma = gamma;
        
        % Laplace
        opts.mode = 'Laplace';
        W_NCRM_Laplace= PCA_NCRM(X,opts);
        H_NCRM_Laplace = W_NCRM_Laplace'* X;
        [ACC_mean_NCRM_Laplace_clu, ACC_std_NCRM_Laplace_clu, NMI_mean_NCRM_Laplace, NMI_std_NCRM_Laplace] = kmeans_for_PCA(H_NCRM_Laplace,gt,k_repeat);
        
        % Geman
        opts.mode = 'Geman';
        W_NCRM_Geman= PCA_NCRM(X,opts);
        H_NCRM_Geman = W_NCRM_Geman'* X;
        [ACC_mean_NCRM_Geman_clu, ACC_std_NCRM_Geman_one_clu, NMI_mean_NCRM_Geman, NMI_std_NCRM_Geman] = kmeans_for_PCA(H_NCRM_Geman,gt,k_repeat);
        
        % ETP
        opts.mode = 'ETP';
        W_NCRM_ETP= PCA_NCRM(X,opts);
        H_NCRM_ETP = W_NCRM_ETP'* X;
        [ACC_mean_NCRM_ETP_clu, ACC_std_NCRM_ETP_clu, NMI_mean_NCRM_ETP, NMI_std_NCRM_ETP] = kmeans_for_PCA(H_NCRM_ETP,gt,k_repeat);
        
        
        % show the clustering results
        fprintf('%20s: ACC_mean = %f, ACC_std = %f, NMI_mean = %f, NMI_std = %f \n', 'PCA_NCRM ETP', ACC_mean_NCRM_ETP_clu, ACC_std_NCRM_ETP_clu, NMI_mean_NCRM_ETP, NMI_std_NCRM_ETP);
        fprintf('%20s: ACC_mean = %f, ACC_std = %f, NMI_mean = %f, NMI_std = %f \n', 'PCA_NCRM Geman', ACC_mean_NCRM_Geman_clu, ACC_std_NCRM_Geman_one_clu, NMI_mean_NCRM_Geman, NMI_std_NCRM_Geman);
        fprintf('%20s: ACC_mean = %f, ACC_std = %f, NMI_mean = %f, NMI_std = %f \n', 'PCA_NCRM Laplace', ACC_mean_NCRM_Laplace_clu, ACC_std_NCRM_Laplace_clu, NMI_mean_NCRM_Laplace, NMI_std_NCRM_Laplace);
        fprintf( '*****************************************************************************************\n');
    end
end




