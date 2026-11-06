clc
clear
close all

%% ================================================================
% Fig. 2 DATA GENERATION ONLY
% Frozen Deep-SSP-HBF tradeoff over the full communications-sensing
% weight region for G = 1,2,4,8.
%
% This file performs NO final paper plotting.
%
% Main purpose:
%   Test whether the SAME Deep-SSP model trained only at
%
%       rho_train = delta_c/delta_s = 1e-3
%       lambda_train = delta_s/delta_c = 1e3
%
%   generalizes across the full SR-CRLB tradeoff region without
%   retraining.
%
% Final deployment setting:
%   parent model depth : 200
%   executed layers    : N_exe = 50
%
% IMPORTANT GENERALIZATION PROTOCOL
% ---------------------------------
% For every weight point, Deep-SSP starts from the SAME deterministic
% channel-specific HBF initialization. There is NO warm start from the
% neighboring weight.
%
% This is deliberate:
%   - the learned gamma/beta schedule is frozen;
%   - the current weight only enters the analytical objective/curvature;
%   - performance across weights therefore measures genuine frozen-model
%     weight generalization rather than continuation across the tradeoff.
%
% The conventional converged SSP-HBF tradeoff should be loaded later by
% a SEPARATE plotting file and used as the reference frontier.
%
% Expected project layout:
%
% Matlab_HBF_ISAC/
% |-- fixed_dataset_K8_Nt64_M3.mat
% |-- DeepSSP_depth_by_G_terminal_gap_g2max3_Fig_data.mat
% `-- Fig2_tradeoff/
%     `-- this_file.m
%
% Output:
% Fig2_tradeoff/results/
%   data_Fig2_DeepSSP_tradeoff_G1_2_4_8_N50_100ch.mat
% ================================================================

%% ================================================================
% Paths
% ================================================================
script_fullpath = mfilename('fullpath');

if isempty(script_fullpath)
    fig_dir = pwd;
else
    fig_dir = fileparts(script_fullpath);
end

project_root = fileparts(fig_dir);

dataset_file = fullfile(project_root, ...
    'fixed_dataset_K8_Nt64_M3.mat');

param_file = fullfile(project_root, ...
    'DeepSSP_depth_by_G_terminal_gap_g2max3_Fig_data.mat');

if ~exist(dataset_file,'file')
    error('Cannot find fixed dataset:\n%s',dataset_file);
end

if ~exist(param_file,'file')
    error('Cannot find trained-model file:\n%s',param_file);
end

Dset = load(dataset_file);
Plearn = load(param_file);

fprintf('============================================================\n');
fprintf('Fig. 2 frozen Deep-SSP tradeoff simulation\n');
fprintf('Dataset    : %s\n',dataset_file);
fprintf('Parameters : %s\n',param_file);
fprintf('============================================================\n');

%% ================================================================
% Required data / parameter fields
% ================================================================
required_data = {'H_test_all','alpha_test','theta_test','phi_test'};

for k = 1:numel(required_data)
    if ~isfield(Dset,required_data{k})
        error('Dataset missing variable "%s".',required_data{k});
    end
end

required_param = {'G_list','gamma_B3','beta_B3','stage1_layers'};

for k = 1:numel(required_param)
    if ~isfield(Plearn,required_param{k})
        error('Parameter file missing variable "%s".',required_param{k});
    end
end

H_test_all = Dset.H_test_all;
alpha_test = Dset.alpha_test;
theta_test = Dset.theta_test;
phi_test   = Dset.phi_test;

G_list = double(Plearn.G_list(:).');
gamma_B3_all = double(Plearn.gamma_B3);
beta_B3_all  = double(Plearn.beta_B3);
stage1_layers = double(Plearn.stage1_layers);

num_G = numel(G_list);

%% ================================================================
% System setting
% ================================================================
M = 3;
K = 8;

Nth = 8;
Ntv = 8;
Nt = Nth * Ntv;

Nrh = 10;
Nrv = 8;
Nr = Nrh * Nrv;

N_RF = K;

Pt      = db2power(20);
noise_c = db2power(0);
noise_s = db2power(0);
L = 64;

I_out = 100;

% Final low-latency execution depth selected from Fig. 1.
N_exe = 50;

% Trained operating point.
rho_train = 1e-3;
lambda_train = 1 / rho_train;   % = 1000

% Fixed HBF projection budget used in training.
hbf_outer_iter = 5;
hbf_inner_iter = 2;

% Deep deployment safeguard.
eps_dec_deep = 1e-8;
use_monotone_accept_deep = true;

% Local-curvature numerical protection.
mu_floor  = 1e-8;
mu_margin = 1e-8;

if size(H_test_all,1) < I_out || ...
        size(alpha_test,1) < I_out || ...
        size(theta_test,1) < I_out || ...
        size(phi_test,1) < I_out

    error('Fixed dataset must contain at least %d test realizations.',I_out);
end

if size(gamma_B3_all,1) ~= num_G || ...
        size(beta_B3_all,1) ~= num_G

    error('gamma_B3/beta_B3 first dimension must match G_list.');
end

if size(gamma_B3_all,2) < N_exe || ...
        size(beta_B3_all,2) < N_exe

    error('Learned parameter arrays need at least %d layers.',N_exe);
end

%% ================================================================
% SAME full tradeoff grid as conventional SSP-HBF Fig. 2
%
% lambda = delta_s / delta_c
% rho    = delta_c / delta_s = 1/lambda
% ================================================================
lambda_list = unique([ ...
    0, ...
    logspace(-4,2,16), ...
    2e2, 3e2, 5e2, 7e2, ...
    1e3, 2e3, 5e3, ...
    1e4, 1e5, 1e6]);

lambda_list = sort(lambda_list,'ascend');

delta_c_list = 1 ./ (1 + lambda_list);
delta_s_list = lambda_list ./ (1 + lambda_list);

rho_list = inf(size(lambda_list));
nz = lambda_list > 0;
rho_list(nz) = 1 ./ lambda_list(nz);

num_weight = numel(lambda_list);

[~,iw_train] = min(abs(lambda_list-lambda_train));

fprintf('\nTradeoff grid:\n');
fprintf('  number of weights = %d\n',num_weight);
fprintf('  lambda range       = [%.1e, %.1e]\n', ...
    min(lambda_list),max(lambda_list));
fprintf('  trained lambda     = %.1f (grid index %d)\n', ...
    lambda_train,iw_train);
fprintf('  trained rho        = %.1e\n',rho_train);
fprintf('  Deep execution     = %d layers\n',N_exe);
fprintf('  channels/point     = %d\n',I_out);
fprintf('  NO warm start across weights.\n\n');

%% ================================================================
% Storage
% Dimensions:
%   [G, channel, weight]
% ================================================================
SR_all   = nan(num_G,I_out,num_weight);
CRLB_all = nan(num_G,I_out,num_weight);
Obj_all  = nan(num_G,I_out,num_weight);

Reject_all = zeros(num_G,I_out,num_weight);
Time_all   = nan(num_G,I_out,num_weight);

completed_weight = false(num_G,num_weight);

%% ================================================================
% Checkpoint / resume
% ================================================================
result_dir = fullfile(fig_dir,'results');

if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

save_file = fullfile(result_dir, ...
    'data_Fig2_DeepSSP_tradeoff_G1_2_4_8_N50_100ch.mat');

resume_existing = true;

if resume_existing && exist(save_file,'file')

    fprintf('Existing checkpoint found:\n%s\n',save_file);

    R = load(save_file);

    compatible = ...
        isfield(R,'G_list') && ...
        isfield(R,'lambda_list') && ...
        isfield(R,'N_exe') && ...
        isequal(double(R.G_list(:).'),G_list) && ...
        isequal(double(R.lambda_list(:).'),lambda_list) && ...
        double(R.N_exe) == N_exe;

    resume_vars = {'SR_all','CRLB_all','Obj_all', ...
                   'Reject_all','Time_all','completed_weight'};

    for k = 1:numel(resume_vars)
        compatible = compatible && isfield(R,resume_vars{k});
    end

    if compatible
        SR_all = R.SR_all;
        CRLB_all = R.CRLB_all;
        Obj_all = R.Obj_all;
        Reject_all = R.Reject_all;
        Time_all = R.Time_all;
        completed_weight = R.completed_weight;

        fprintf('Resume accepted: %d/%d G-weight points completed.\n\n', ...
            nnz(completed_weight),numel(completed_weight));
    else
        warning('Existing checkpoint incompatible; starting from scratch.');
    end
end

%% ================================================================
% Main loops
%
% C2S order = increasing lambda, but every weight is initialized
% independently; run order therefore does not change the solution.
% ================================================================
for iG = 1:num_G

    G = G_list(iG);
    Z = build_groupconnected_mask(Nt,N_RF,G);

    gamma_B3 = gamma_B3_all(iG,1:N_exe);
    beta_B3  = beta_B3_all(iG,1:N_exe);

    fprintf('\n============================================================\n');
    fprintf('G = %d (%d/%d)\n',G,iG,num_G);
    fprintf('============================================================\n');

    for iw = 1:num_weight

        if completed_weight(iG,iw)
            fprintf('G=%d, weight %2d/%2d already completed, skip.\n', ...
                G,iw,num_weight);
            continue;
        end

        lambda = lambda_list(iw);
        delta_c = delta_c_list(iw);
        delta_s = delta_s_list(iw);

        fprintf('\n------------------------------------------------------------\n');
        fprintf(['G=%d | weight %2d/%2d | lambda=%.3e | ' ...
                 'delta_c=%.3e | delta_s=%.3e%s\n'], ...
            G,iw,num_weight,lambda,delta_c,delta_s, ...
            ternary_text(iw==iw_train,'  <-- training weight',''));
        fprintf('------------------------------------------------------------\n');

        for channel = 1:I_out

            if mod(channel,10)==1 || channel==I_out
                fprintf('  channel %3d/%3d\n',channel,I_out);
            end

            H = double(squeeze(H_test_all(channel,:,1:K)));
            alpha = double(alpha_test(channel,1:M).');
            theta = double(theta_test(channel,1:M).');
            phi   = double(phi_test(channel,1:M).');

            %% Steering matrices
            [A,dAtheta,dAphi] = ...
                construct_steer_matrix_and_derivative_steer_matrix( ...
                    theta,phi,Nth,Ntv);

            [B,dBtheta,dBphi] = ...
                construct_steer_matrix_and_derivative_steer_matrix( ...
                    theta,phi,Nrh,Nrv);

            Gamma = diag(alpha);

            %% Precompute sensing terms
            Pre = struct();

            Pre.tilde_A    = A';
            Pre.tilde_A_o  = Gamma * A';
            Pre.tilde_A_th = Gamma * dAtheta';
            Pre.tilde_A_ph = Gamma * dAphi';

            Pre.tilde_B_oo   = B'       * B;
            Pre.tilde_B_oth  = B'       * dBtheta;
            Pre.tilde_B_oph  = B'       * dBphi;
            Pre.tilde_B_tho  = dBtheta' * B;
            Pre.tilde_B_thth = dBtheta' * dBtheta;
            Pre.tilde_B_thph = dBtheta' * dBphi;
            Pre.tilde_B_pho  = dBphi'   * B;
            Pre.tilde_B_phph = dBphi'   * dBphi;

            Pre.L = L;
            Pre.noise_s = noise_s;
            Pre.gamma = alpha;
            Pre.Gamma = Gamma;

            %% Compact basis
            Nbasis = [H,A,dAtheta,dAphi];
            [~,R_N,~] = qr_rank_basis(Nbasis);

            %% SAME deterministic initialization for every weight
            Wa = Z .* exp(1j * angle(H(:,1:N_RF)));

            Wd = digital_update_groupconnected(Wa,H(:,1:K));
            Wd = sqrt(Pt) / max(norm(Wa*Wd,'fro'),1e-12) * Wd;

            W = Wa * Wd;

            [obj_cur,sr_cur,crlb_cur,FIM] = ...
                evaluate_metrics( ...
                    Pre,H,W,delta_c,delta_s,noise_c,K);

            W_prev_layer = W;
            reject_count = 0;

            tstart = tic;

            %% ====================================================
            % Frozen Deep-SSP, N_exe = 50
            % =====================================================
            for t = 1:N_exe

                W_old = W;
                Wa_old = Wa;
                Wd_old = Wd;
                obj_old = obj_cur;

                [grad_total,lambda_local] = ...
                    compute_grad_and_local_curvature( ...
                        Pre,H,W_old,FIM,Nbasis,R_N, ...
                        delta_c,delta_s,noise_c,K,M, ...
                        mu_floor,mu_margin);

                gamma_t = gamma_B3(t);

                if t <= stage1_layers
                    beta_t = 0;
                else
                    beta_t = beta_B3(t);
                end

                inertial_state = ...
                    W_old + beta_t * (W_old-W_prev_layer);

                effective_shift = ...
                    lambda_local / max(gamma_t,1e-8);

                % Exact scaled target matching the trained PyTorch model.
                W_target = ...
                    grad_total + ...
                    effective_shift * inertial_state;

                [Wa_cand,Wd_cand] = ...
                    matrix_approximation_groupconnected_fixed( ...
                        W_target,Wa_old,Wd_old,Z, ...
                        hbf_outer_iter,hbf_inner_iter);

                Wd_cand = ...
                    sqrt(Pt) / ...
                    max(norm(Wa_cand*Wd_cand,'fro'),1e-12) * ...
                    Wd_cand;

                W_cand = Wa_cand * Wd_cand;

                [obj_cand,sr_cand,crlb_cand,FIM_cand] = ...
                    evaluate_metrics( ...
                        Pre,H,W_cand, ...
                        delta_c,delta_s,noise_c,K);

                if ~use_monotone_accept_deep || ...
                        obj_cand >= obj_old-eps_dec_deep

                    W_prev_layer = W_old;

                    W = W_cand;
                    Wa = Wa_cand;
                    Wd = Wd_cand;
                    FIM = FIM_cand;

                    obj_cur = obj_cand;
                    sr_cur = sr_cand;
                    crlb_cur = crlb_cand;

                else
                    % Reject candidate and retain current point.
                    % Next displacement is therefore zero.
                    W_prev_layer = W_old;
                    W = W_old;
                    reject_count = reject_count + 1;
                end
            end

            Time_all(iG,channel,iw) = toc(tstart);

            SR_all(iG,channel,iw) = sr_cur;
            CRLB_all(iG,channel,iw) = crlb_cur;
            Obj_all(iG,channel,iw) = obj_cur;

            Reject_all(iG,channel,iw) = reject_count;
        end

        completed_weight(iG,iw) = true;

        %% Current averages
        SR_avg = squeeze(mean(SR_all,2,'omitnan'));
        CRLB_avg = squeeze(mean(CRLB_all,2,'omitnan'));
        Obj_avg = squeeze(mean(Obj_all,2,'omitnan'));
        Time_avg = squeeze(mean(Time_all,2,'omitnan'));
        Reject_avg = squeeze(mean(Reject_all,2,'omitnan'));

        save(save_file, ...
            'G_list','lambda_list','rho_list', ...
            'delta_c_list','delta_s_list', ...
            'rho_train','lambda_train','iw_train', ...
            'M','K','Nt','Nr','N_RF','Nth','Ntv','Nrh','Nrv', ...
            'Pt','noise_c','noise_s','L','I_out', ...
            'N_exe','stage1_layers', ...
            'hbf_outer_iter','hbf_inner_iter', ...
            'gamma_B3_all','beta_B3_all', ...
            'SR_all','CRLB_all','Obj_all','Time_all','Reject_all', ...
            'SR_avg','CRLB_avg','Obj_avg','Time_avg','Reject_avg', ...
            'completed_weight','-v7.3');

        fprintf(['Completed G=%d, lambda=%.3e | avg SR=%.5f | ' ...
                 'avg CRLB=%.6e | avg rejects=%.2f\n'], ...
            G,lambda, ...
            SR_avg(iG,iw), ...
            CRLB_avg(iG,iw), ...
            Reject_avg(iG,iw));
    end
end

%% ================================================================
% Final averages and quick consistency output
% ================================================================
SR_avg = squeeze(mean(SR_all,2,'omitnan'));
CRLB_avg = squeeze(mean(CRLB_all,2,'omitnan'));
Obj_avg = squeeze(mean(Obj_all,2,'omitnan'));
Time_avg = squeeze(mean(Time_all,2,'omitnan'));
Reject_avg = squeeze(mean(Reject_all,2,'omitnan'));

save(save_file, ...
    'G_list','lambda_list','rho_list', ...
    'delta_c_list','delta_s_list', ...
    'rho_train','lambda_train','iw_train', ...
    'M','K','Nt','Nr','N_RF','Nth','Ntv','Nrh','Nrv', ...
    'Pt','noise_c','noise_s','L','I_out', ...
    'N_exe','stage1_layers', ...
    'hbf_outer_iter','hbf_inner_iter', ...
    'gamma_B3_all','beta_B3_all', ...
    'SR_all','CRLB_all','Obj_all','Time_all','Reject_all', ...
    'SR_avg','CRLB_avg','Obj_avg','Time_avg','Reject_avg', ...
    'completed_weight','-v7.3');

fprintf('\n============================================================\n');
fprintf('Deep-SSP tradeoff simulation complete.\n');
fprintf('Saved to:\n%s\n',save_file);
fprintf('Frozen training weight: lambda=%.1f, rho=%.1e\n', ...
    lambda_train,rho_train);
fprintf('Execution depth: N_exe=%d\n',N_exe);
fprintf('============================================================\n');

for iG = 1:num_G
    fprintf(['G=%d at training weight: SR=%.5f, CRLB=%.6e, ' ...
             'rejects=%.2f\n'], ...
        G_list(iG), ...
        SR_avg(iG,iw_train), ...
        CRLB_avg(iG,iw_train), ...
        Reject_avg(iG,iw_train));
end

%% =================================================================
% Local functions
% =================================================================
function [grad_total,lambda_local] = ...
    compute_grad_and_local_curvature( ...
        Pre,H,W,FIM,Nbasis,R_N,delta_c,delta_s, ...
        noise_c,K,M,mu_floor,mu_margin)

    T_k = sum(square_abs(H'*W),2) + ...
        noise_c*ones(K,1);

    alpha_k = ...
        T_k ./ ...
        (T_k-square_abs(diag(H'*W))) - 1;

    beta_k = ...
        sqrt(1+alpha_k) .* ...
        diag(H'*W) ./ T_k;

    Sigma1 = ...
        diag(sqrt(1+alpha_k).*beta_k);

    Sigma2 = ...
        diag(square_abs(beta_k));

    CRBM = inv(FIM);
    Phi = CRBM*CRBM;

    Sigma3 = construct_Sigma3(Pre,Phi);

    Omega1 = ...
        [delta_c*Sigma1; zeros(3*M,K)];

    Omega2 = ...
        blkdiag(-delta_c*Sigma2,delta_s*Sigma3);

    [lambda_exact,~] = ...
        compute_lowdim_exact_shift_from_Omega2( ...
            R_N,Omega2);

    lambda_local = ...
        max(lambda_exact+mu_margin,mu_floor);

    grad_total = ...
        Nbasis * ...
        (Omega1 + Omega2*(Nbasis'*W));
end


function [obj,sr,crlb,FIM] = ...
    evaluate_metrics( ...
        Pre,H,W,delta_c,delta_s,noise_c,K)

    FIM = calculateFIM(Pre,W);

    crlb = trace(inv(FIM));

    T_k = ...
        sum(square_abs(H'*W),2) + ...
        noise_c*ones(K,1);

    rate_p = ...
        log(T_k ./ ...
        (T_k-square_abs(diag(H'*W))));

    sr = sum(rate_p);

    obj = delta_c*sr-delta_s*crlb;
end


function [mu_exact,eig_red] = ...
    compute_lowdim_exact_shift_from_Omega2( ...
        R_N,Omega2)

    C2_red = R_N*Omega2*R_N';
    C2_red = (C2_red+C2_red')/2;

    eig_red = real(eig(C2_red));

    if isempty(eig_red)
        mu_exact = 0;
    else
        mu_exact = max(abs(eig_red));
    end
end


function [U,R_basis,r] = qr_rank_basis(X)

    [Q,R] = qr(X,0);

    if isempty(R)
        U = Q;
        R_basis = R;
        r = 0;
        return;
    end

    tol_rank = ...
        max(size(X))*eps(class(X))* ...
        max(abs(diag(R)));

    r = sum(abs(diag(R))>tol_rank);

    U = Q(:,1:r);
    R_basis = R(1:r,:);
end


function Sigma3 = construct_Sigma3(Pre,Phi)

    M = size(Pre.tilde_B_oo,1);

    phi11 = Phi(1:M,1:M);
    phi12 = Phi(1:M,M+1:2*M);
    phi13 = Phi(1:M,2*M+1:3*M);
    phi14 = Phi(1:M,3*M+1:4*M);

    phi22 = Phi(M+1:2*M,M+1:2*M);
    phi23 = Phi(M+1:2*M,2*M+1:3*M);
    phi24 = Phi(M+1:2*M,3*M+1:4*M);

    phi33 = Phi(2*M+1:3*M,2*M+1:3*M);
    phi34 = Phi(2*M+1:3*M,3*M+1:4*M);

    phi44 = Phi(3*M+1:4*M,3*M+1:4*M);

    Boo   = Pre.tilde_B_oo;
    Both  = Pre.tilde_B_oth;
    Boph  = Pre.tilde_B_oph;
    Btho  = Pre.tilde_B_tho;
    Bthth = Pre.tilde_B_thth;
    Bthph = Pre.tilde_B_thph;
    Bpho  = Pre.tilde_B_pho;
    Bphph = Pre.tilde_B_phph;

    Moo = ...
        phi11.*Bthth + ...
        2*phi12.*Bthph + ...
        phi22.*Bphph;

    Mtho  = 2*(phi11.*Both + phi12.*Boph);
    Mpho  = 2*(phi22.*Boph);
    Moph  = 2*(phi12.*Btho);
    Mthph = 2*(phi12.*Boo);
    Mthth = phi11.*Boo;
    Mphph = phi22.*Boo;

    Mo3 = ...
        (2*phi13+2j*phi14).*Btho + ...
        (2*phi23+2j*phi24).*Bpho;

    Mth3 = ...
        (2*phi13+2j*phi14).*Boo;

    Mph3 = ...
        (2*phi23+2j*phi24).*Boo;

    M33 = ...
        (phi33+phi44+2j*phi34).*Boo;

    Gamma = Pre.Gamma;

    idxA  = 1:M;
    idxTh = M+1:2*M;
    idxPh = 2*M+1:3*M;

    Sigma3_tilde = zeros(3*M,3*M);

    Sigma3_tilde(idxA,idxA) = ...
        Sigma3_tilde(idxA,idxA) + ...
        Gamma'*Moo*Gamma;

    Sigma3_tilde(idxTh,idxA) = ...
        Sigma3_tilde(idxTh,idxA) + ...
        Gamma'*Mtho*Gamma;

    Sigma3_tilde(idxPh,idxA) = ...
        Sigma3_tilde(idxPh,idxA) + ...
        Gamma'*Mpho*Gamma;

    Sigma3_tilde(idxA,idxPh) = ...
        Sigma3_tilde(idxA,idxPh) + ...
        Gamma'*Moph*Gamma;

    Sigma3_tilde(idxTh,idxPh) = ...
        Sigma3_tilde(idxTh,idxPh) + ...
        Gamma'*Mthph*Gamma;

    Sigma3_tilde(idxPh,idxPh) = ...
        Sigma3_tilde(idxPh,idxPh) + ...
        Gamma'*Mphph*Gamma;

    Sigma3_tilde(idxTh,idxTh) = ...
        Sigma3_tilde(idxTh,idxTh) + ...
        Gamma'*Mthth*Gamma;

    Sigma3_tilde(idxA,idxA) = ...
        Sigma3_tilde(idxA,idxA) + ...
        Gamma'*Mo3;

    Sigma3_tilde(idxTh,idxA) = ...
        Sigma3_tilde(idxTh,idxA) + ...
        Gamma'*Mth3;

    Sigma3_tilde(idxPh,idxA) = ...
        Sigma3_tilde(idxPh,idxA) + ...
        Gamma'*Mph3;

    Sigma3_tilde(idxA,idxA) = ...
        Sigma3_tilde(idxA,idxA) + M33;

    Sigma3 = ...
        (2*Pre.L/Pre.noise_s) * ...
        (Sigma3_tilde+Sigma3_tilde')/2;
end


function FIM = calculateFIM(Pre,D)

    Wo  = Pre.tilde_A_o  * D;
    Wth = Pre.tilde_A_th * D;
    Wph = Pre.tilde_A_ph * D;
    Wa  = Pre.tilde_A    * D;

    Goo   = Wo*Wo';
    Goth  = Wo*Wth';
    Gtho  = Wth*Wo';
    Goph  = Wo*Wph';
    Gpho  = Wph*Wo';
    Gthth = Wth*Wth';
    Gphph = Wph*Wph';
    Gath  = Wa*Wth';
    Gao   = Wa*Wo';
    Gaph  = Wa*Wph';
    Gaa   = Wa*Wa';
    Gphth = Wph*Wth';

    F11 = ...
        (Goo).'   .* Pre.tilde_B_thth + ...
        (Goth).'  .* Pre.tilde_B_oth  + ...
        (Gtho).'  .* Pre.tilde_B_tho  + ...
        (Gthth).' .* Pre.tilde_B_oo;

    F12 = ...
        (Goo).'   .* Pre.tilde_B_thph + ...
        (Goth).'  .* Pre.tilde_B_oph  + ...
        (Gpho).'  .* Pre.tilde_B_tho  + ...
        (Gphth).' .* Pre.tilde_B_oo;

    F13 = ...
        (Gao).'  .* Pre.tilde_B_tho + ...
        (Gath).' .* Pre.tilde_B_oo;

    F22 = ...
        (Goo).'   .* Pre.tilde_B_phph + ...
        (Goph).'  .* Pre.tilde_B_oph  + ...
        (Gpho).'  .* Pre.tilde_B_pho  + ...
        (Gphph).' .* Pre.tilde_B_oo;

    F23 = ...
        (Gao).'  .* Pre.tilde_B_pho + ...
        (Gaph).' .* Pre.tilde_B_oo;

    F33 = ...
        (Gaa).' .* Pre.tilde_B_oo;

    M = size(Pre.tilde_B_oo,1);

    FIM = zeros(4*M);

    FIM(1:M,1:M) = real(F11);
    FIM(1:M,M+1:2*M) = real(F12);
    FIM(1:M,2*M+1:3*M) = real(F13);
    FIM(1:M,3*M+1:4*M) = -imag(F13);

    FIM(M+1:2*M,M+1:2*M) = real(F22);
    FIM(M+1:2*M,2*M+1:3*M) = real(F23);
    FIM(M+1:2*M,3*M+1:4*M) = -imag(F23);

    FIM(2*M+1:3*M,2*M+1:3*M) = real(F33);
    FIM(2*M+1:3*M,3*M+1:4*M) = -imag(F33);

    FIM(3*M+1:4*M,3*M+1:4*M) = real(F33);

    FIM = triu(FIM) + triu(FIM,1)';

    FIM = 2*Pre.L/Pre.noise_s*FIM;
end


function [A,dAtheta,dAphi] = ...
    construct_steer_matrix_and_derivative_steer_matrix( ...
        theta,phi,Mx,My)

    ix = (0:Mx-1)';
    iy = (0:My-1)';

    M = length(theta);

    A = zeros(Mx*My,M);
    dAtheta = zeros(Mx*My,M);
    dAphi = zeros(Mx*My,M);

    for m = 1:M

        ax = ...
            1/sqrt(Mx) * ...
            exp(1j*pi*ix*sin(theta(m))*sin(phi(m)));

        ay = ...
            1/sqrt(My) * ...
            exp(1j*pi*iy*cos(phi(m)));

        daxtheta = ...
            1j*pi*ix*cos(theta(m))*sin(phi(m)).*ax;

        daxphi = ...
            1j*pi*ix*sin(theta(m))*cos(phi(m)).*ax;

        dayphi = ...
            -1j*pi*iy*sin(phi(m)).*ay;

        A(:,m) = kron(ay,ax);

        dAtheta(:,m) = ...
            kron(ay,daxtheta);

        dAphi(:,m) = ...
            kron(ay,daxphi) + ...
            kron(dayphi,ax);
    end
end


function [Wa,Wd] = ...
    matrix_approximation_groupconnected_fixed( ...
        Wtarget,Wa,Wd,Z,outer_iter,inner_iter)

    Wa = Z .* exp(1j*angle(Wa));

    for count = 1:outer_iter

        Wd = digital_update_groupconnected(Wa,Wtarget);

        Rwd = Wd*Wd';
        Rwd = (Rwd+Rwd')/2;

        mu1 = gershgorin_shift(Rwd);

        for ii = 1:inner_iter

            D = ...
                -Wa*Rwd + ...
                Wtarget*Wd' + ...
                mu1*Wa;

            Wa = Z .* exp(1j*angle(D));
        end
    end
end


function mu = gershgorin_shift(C)

    diag_abs = abs(diag(C));

    row_sum = ...
        sum(abs(C),2)-diag_abs;

    diag_real = real(diag(C));

    mu = ...
        max(diag_real+row_sum)+1e-6;
end


function Wd = ...
    digital_update_groupconnected(Wa,Wtarget)

    Wd = Wa\Wtarget;
end


function Z = ...
    build_groupconnected_mask(Nt,N_RF,G)

    if G < 1 || ...
            G > min(Nt,N_RF) || ...
            round(G) ~= G

        error(['G must be an integer satisfying ' ...
               '1 <= G <= min(Nt,N_RF).']);
    end

    if mod(Nt,G) ~= 0 || ...
            mod(N_RF,G) ~= 0

        error('G must divide Nt and N_RF.');
    end

    Nant_g = Nt/G;
    Nrf_g  = N_RF/G;

    Z = zeros(Nt,N_RF);

    for gg = 1:G

        ant_idx = ...
            (gg-1)*Nant_g + (1:Nant_g);

        rf_idx = ...
            (gg-1)*Nrf_g + (1:Nrf_g);

        Z(ant_idx,rf_idx) = 1;
    end
end


function y = db2power(x)
    y = 10.^(x/10);
end


function y = square_abs(x)
    y = abs(x).^2;
end


function out = ternary_text(cond,a,b)

    if cond
        out = a;
    else
        out = b;
    end
end
