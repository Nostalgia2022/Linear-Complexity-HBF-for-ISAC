clc
clear
close all

%% ========================================================================
% Fig. 5 DATA GENERATION ONLY
% Frozen Deep-SSP-HBF generalization over:
%   (a) transmit power Pt
%   (b) number of transmit antennas Nt
%
% Nominal training configuration:
%   Nt = 64 (8x8 UPA), Nr = 80 (10x8 UPA)
%   K = 8, M = 3, N_RF = 8, G = 4
%   Pt = 20 dBm, rho = delta_c/delta_s = 1e-3
%
% Frozen model:
%   final G=4 parent model, Nmax = 200
%
% Methods saved:
%   1) SSP-HBF, N = 50
%   2) Deep-SSP-HBF, N = 50
%   3) Deep-SSP-HBF, N = 200     [diagnostic]
%   4) Long-run SSP-HBF, <= 2000 iterations [reference]
%
% IMPORTANT:
%   * plotting is separate;
%   * the learned scalar schedule is frozen for ALL Pt and Nt;
%   * Pt sweep reuses exactly the shared nominal fixed dataset;
%   * Nt sweep uses the SAME target samples as the nominal dataset and a
%     deterministic nested Rayleigh channel realization generated at Nt=128,
%     so Nt=32/64/128 use the first Nt rows of the SAME underlying channel;
%   * G=4 and N_RF=8 remain fixed;
%   * checkpoint/resume is supported.
%
% Expected project layout:
%
% Matlab_HBF_ISAC/
% |-- fixed_dataset_K8_Nt64_M3.mat
% |-- DeepSSP_depth_by_G_terminal_gap_g2max3_Fig_data.mat
% `-- Fig5_generalization_Pt_Nt/
%     `-- this_file.m
%
% Output:
% Fig5_generalization_Pt_Nt/results/
%   data_Fig5_DeepSSP_generalization_Pt_Nt_G4_100ch.mat
%% ========================================================================

%% Paths
script_fullpath = mfilename('fullpath');
if isempty(script_fullpath)
    fig_dir = pwd;
else
    fig_dir = fileparts(script_fullpath);
end
project_root = fileparts(fig_dir);

dataset_file = fullfile(project_root,'fixed_dataset_K8_Nt64_M3.mat');
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

%% Nominal setting
K = 8;
M = 3;

Nrh = 10;
Nrv = 8;
Nr = Nrh*Nrv;

N_RF = 8;
G = 4;

Pt0_dBm = 20;
Pt0 = db2power(Pt0_dBm);

noise_c = db2power(0);
noise_s = db2power(0);
L = 64;

rho0 = 1e-3;
delta_c0 = rho0/(1+rho0);
delta_s0 = 1/(1+rho0);

I_out = 100;

%% Frozen G=4 schedule
G_learn = double(Plearn.G_list(:).');
idxG = find(G_learn==G,1,'first');
if isempty(idxG)
    error('G=%d not found in trained parameter file.',G);
end

gamma_all = double(Plearn.gamma_B3);
beta_all  = double(Plearn.beta_B3);
stage1_layers = double(Plearn.stage1_layers);

Nexe = 50;
Ndeep = 200;

if size(gamma_all,2)<Ndeep || size(beta_all,2)<Ndeep
    error('Learned schedule contains fewer than %d layers.',Ndeep);
end

gamma_B3 = gamma_all(idxG,1:Ndeep);
beta_B3  = beta_all(idxG,1:Ndeep);

%% Solver settings
reference_iter = 2000;
tol = 1e-5;

hbf_outer_iter = 5;
hbf_inner_iter = 2;

max_accept_trials = 20;
rho_in = 2;

b1_accept_tol = 1e-10;
deep_accept_tol = 1e-8;

mu_floor = 1e-8;
mu_margin = 1e-8;

%% Sweeps
Pt_dBm_list = 0:5:30;
Pt_list = db2power(Pt_dBm_list);

% Keep N_RF=8 and G=4 feasible for every Nt.
Nt_list = [32 48 64 80 96 112 128];

% UPA dimensions: keep 8 vertical elements and vary the horizontal size.
% The nominal training array 8x8 (Nt=64) is included in the middle.
Nth_list = [4 6 8 10 12 14 16];
Ntv_list = 8*ones(size(Nth_list));

if any(Nth_list.*Ntv_list ~= Nt_list)
    error('Nth_list .* Ntv_list must equal Nt_list.');
end

if any(mod(Nt_list,G)~=0) || mod(N_RF,G)~=0
    error('Every Nt and N_RF must be divisible by G.');
end

fprintf('============================================================\n');
fprintf('Fig. 5: frozen Deep-SSP generalization over Pt and Nt\n');
fprintf('Nominal: Pt=%d dBm, Nt=64, K=%d, M=%d, G=%d\n', ...
    Pt0_dBm,K,M,G);
fprintf('Pt sweep: [%s] dBm\n',num2str(Pt_dBm_list));
fprintf('Nt sweep: [%s]\n',num2str(Nt_list));
fprintf('Frozen depths: Nexe=%d, Ndeep=%d\n',Nexe,Ndeep);
fprintf('============================================================\n');

%% ========================================================================
% Deterministic nested Nt-sweep channels
%
% Generate H128 once. Nt=32/64/128 take first Nt antenna rows.
% This gives paired channel randomness across array sizes.
%% ========================================================================
Nt_max = max(Nt_list);

rng(9202026,'twister');
H_Nt_nested = 1/sqrt(2) * ...
    (randn(I_out,Nt_max,K) + 1j*randn(I_out,Nt_max,K));

% Targets exactly reuse nominal shared dataset.
alpha_nom = double(alpha_test(1:I_out,1:M));
theta_nom = double(theta_test(1:I_out,1:M));
phi_nom   = double(phi_test(1:I_out,1:M));

%% Storage
Ptres = init_sweep_storage(numel(Pt_list),I_out);
Ntres = init_sweep_storage(numel(Nt_list),I_out);

done_Pt = false(numel(Pt_list),I_out);
done_Nt = false(numel(Nt_list),I_out);

result_dir = fullfile(fig_dir,'results');
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

save_file = fullfile(result_dir, ...
    'data_Fig5_DeepSSP_generalization_Pt_Nt_G4_100ch.mat');

%% Resume
resume_existing = true;

if resume_existing && exist(save_file,'file')

    R = load(save_file);

    compatible = ...
        isfield(R,'Pt_dBm_list') && isfield(R,'Nt_list') && ...
        isfield(R,'Ptres') && isfield(R,'Ntres') && ...
        isfield(R,'done_Pt') && isfield(R,'done_Nt') && ...
        isequal(double(R.Pt_dBm_list(:).'),Pt_dBm_list) && ...
        isequal(double(R.Nt_list(:).'),Nt_list);

    if compatible
        Ptres = R.Ptres;
        Ntres = R.Ntres;
        done_Pt = R.done_Pt;
        done_Nt = R.done_Nt;

        fprintf('Resume accepted: Pt %d/%d, Nt %d/%d cases complete.\n', ...
            nnz(done_Pt),numel(done_Pt),nnz(done_Nt),numel(done_Nt));
    else
        warning('Checkpoint incompatible; starting from scratch.');
    end
end

%% ========================================================================
% (a) Pt sweep -- exactly shared nominal Nt=64 dataset
%% ========================================================================
fprintf('\n================ Pt SWEEP ================\n');

Nth = 8;
Ntv = 8;
Nt = 64;
Z = build_groupconnected_mask(Nt,N_RF,G);

for iP = 1:numel(Pt_list)

    Pt = Pt_list(iP);
    Pt_dBm = Pt_dBm_list(iP);

    fprintf('\n---- Pt = %g dBm (%d/%d) ----\n', ...
        Pt_dBm,iP,numel(Pt_list));

    for channel = 1:I_out

        if done_Pt(iP,channel)
            continue;
        end

        H = double(squeeze(H_test_all(channel,:,1:K)));
        H_seed = H;

        alpha = double(alpha_test(channel,1:M).');
        theta = double(theta_test(channel,1:M).');
        phi   = double(phi_test(channel,1:M).');

        out = evaluate_generalization_case( ...
            H,H_seed,alpha,theta,phi, ...
            Nth,Ntv,Nrh,Nrv,Z, ...
            Pt,noise_c,noise_s,L, ...
            delta_c0,delta_s0, ...
            gamma_B3,beta_B3,stage1_layers,Nexe,Ndeep, ...
            reference_iter,tol, ...
            hbf_outer_iter,hbf_inner_iter, ...
            max_accept_trials,rho_in, ...
            b1_accept_tol,deep_accept_tol, ...
            mu_floor,mu_margin);

        Ptres = store_case_result(Ptres,iP,channel,out);
        done_Pt(iP,channel) = true;

        if channel==1 || mod(channel,10)==0 || channel==I_out
            fprintf(['Pt=%g dBm, ch %3d/%3d | gaps: ' ...
                     'SSP50=%.4f, Deep50=%.4f, Deep200=%.4f\n'], ...
                Pt_dBm,channel,I_out, ...
                out.gap_B1_50,out.gap_Deep50,out.gap_Deep200);
        end

        if mod(channel,5)==0 || channel==I_out
            save_checkpoint(save_file, ...
                Pt_dBm_list,Pt_list,Nt_list,Nth_list,Ntv_list, ...
                Ptres,Ntres,done_Pt,done_Nt,H_Nt_nested, ...
                K,M,Nr,N_RF,G,Pt0_dBm,Pt0,noise_c,noise_s,L,rho0, ...
                delta_c0,delta_s0,Nexe,Ndeep,stage1_layers, ...
                gamma_B3,beta_B3);
        end
    end
end

%% ========================================================================
% (b) Nt sweep -- paired nested channels, Pt fixed at 20 dBm
%% ========================================================================
fprintf('\n================ Nt SWEEP ================\n');

for iN = 1:numel(Nt_list)

    Nt = Nt_list(iN);
    Nth = Nth_list(iN);
    Ntv = Ntv_list(iN);

    Z = build_groupconnected_mask(Nt,N_RF,G);

    fprintf('\n---- Nt = %d (%dx%d) (%d/%d) ----\n', ...
        Nt,Nth,Ntv,iN,numel(Nt_list));

    for channel = 1:I_out

        if done_Nt(iN,channel)
            continue;
        end

        H = squeeze(H_Nt_nested(channel,1:Nt,1:K));
        H_seed = H;

        alpha = alpha_nom(channel,:).';
        theta = theta_nom(channel,:).';
        phi   = phi_nom(channel,:).';

        out = evaluate_generalization_case( ...
            H,H_seed,alpha,theta,phi, ...
            Nth,Ntv,Nrh,Nrv,Z, ...
            Pt0,noise_c,noise_s,L, ...
            delta_c0,delta_s0, ...
            gamma_B3,beta_B3,stage1_layers,Nexe,Ndeep, ...
            reference_iter,tol, ...
            hbf_outer_iter,hbf_inner_iter, ...
            max_accept_trials,rho_in, ...
            b1_accept_tol,deep_accept_tol, ...
            mu_floor,mu_margin);

        Ntres = store_case_result(Ntres,iN,channel,out);
        done_Nt(iN,channel) = true;

        if channel==1 || mod(channel,10)==0 || channel==I_out
            fprintf(['Nt=%d, ch %3d/%3d | gaps: ' ...
                     'SSP50=%.4f, Deep50=%.4f, Deep200=%.4f\n'], ...
                Nt,channel,I_out, ...
                out.gap_B1_50,out.gap_Deep50,out.gap_Deep200);
        end

        if mod(channel,5)==0 || channel==I_out
            save_checkpoint(save_file, ...
                Pt_dBm_list,Pt_list,Nt_list,Nth_list,Ntv_list, ...
                Ptres,Ntres,done_Pt,done_Nt,H_Nt_nested, ...
                K,M,Nr,N_RF,G,Pt0_dBm,Pt0,noise_c,noise_s,L,rho0, ...
                delta_c0,delta_s0,Nexe,Ndeep,stage1_layers, ...
                gamma_B3,beta_B3);
        end
    end
end

%% Finalize
Ptres = finalize_sweep_storage(Ptres);
Ntres = finalize_sweep_storage(Ntres);

save_checkpoint(save_file, ...
    Pt_dBm_list,Pt_list,Nt_list,Nth_list,Ntv_list, ...
    Ptres,Ntres,done_Pt,done_Nt,H_Nt_nested, ...
    K,M,Nr,N_RF,G,Pt0_dBm,Pt0,noise_c,noise_s,L,rho0, ...
    delta_c0,delta_s0,Nexe,Ndeep,stage1_layers, ...
    gamma_B3,beta_B3);

fprintf('\n============================================================\n');
fprintf('Fig. 5 Pt/Nt generalization simulation complete.\n');
fprintf('Saved to:\n%s\n',save_file);

fprintf('\nPt sweep:\n');
fprintf('Pt(dBm) | SSP50 gap | Deep50 gap | Deep200 gap\n');
for i = 1:numel(Pt_dBm_list)
    fprintf('%7.1f | %9.5f | %10.5f | %11.5f\n', ...
        Pt_dBm_list(i), ...
        Ptres.Gap_B1_50_avg(i), ...
        Ptres.Gap_Deep50_avg(i), ...
        Ptres.Gap_Deep200_avg(i));
end

fprintf('\nNt sweep:\n');
fprintf(' Nt | SSP50 gap | Deep50 gap | Deep200 gap\n');
for i = 1:numel(Nt_list)
    fprintf('%3d | %9.5f | %10.5f | %11.5f\n', ...
        Nt_list(i), ...
        Ntres.Gap_B1_50_avg(i), ...
        Ntres.Gap_Deep50_avg(i), ...
        Ntres.Gap_Deep200_avg(i));
end
fprintf('============================================================\n');

%% ========================================================================
% Local functions
%% ========================================================================
function out = evaluate_generalization_case( ...
    H,H_seed,alpha,theta,phi, ...
    Nth,Ntv,Nrh,Nrv,Z, ...
    Pt,noise_c,noise_s,L,delta_c,delta_s, ...
    gamma_B3,beta_B3,stage1_layers,Nexe,Ndeep, ...
    reference_iter,tol, ...
    hbf_outer_iter,hbf_inner_iter, ...
    max_accept_trials,rho_in, ...
    b1_accept_tol,deep_accept_tol, ...
    mu_floor,mu_margin)

    K = size(H,2);
    M = numel(theta);
    N_RF = size(Z,2);

    [A,dAtheta,dAphi] = ...
        construct_steer_matrix_and_derivative_steer_matrix(theta,phi,Nth,Ntv);

    [B,dBtheta,dBphi] = ...
        construct_steer_matrix_and_derivative_steer_matrix(theta,phi,Nrh,Nrv);

    Gamma = diag(alpha);

    Pre = struct();
    Pre.tilde_A    = A';
    Pre.tilde_A_o  = Gamma*A';
    Pre.tilde_A_th = Gamma*dAtheta';
    Pre.tilde_A_ph = Gamma*dAphi';

    Pre.tilde_B_oo   = B'*B;
    Pre.tilde_B_oth  = B'*dBtheta;
    Pre.tilde_B_oph  = B'*dBphi;
    Pre.tilde_B_tho  = dBtheta'*B;
    Pre.tilde_B_thth = dBtheta'*dBtheta;
    Pre.tilde_B_thph = dBtheta'*dBphi;
    Pre.tilde_B_pho  = dBphi'*B;
    Pre.tilde_B_phph = dBphi'*dBphi;

    Pre.L = L;
    Pre.noise_s = noise_s;
    Pre.gamma = alpha;
    Pre.Gamma = Gamma;

    Nbasis = [H,A,dAtheta,dAphi];
    [~,R_N,~] = qr_rank_basis(Nbasis);

    if size(H_seed,2)<N_RF
        error('H_seed must contain at least N_RF columns.');
    end

    Wa0 = Z.*exp(1j*angle(H_seed(:,1:N_RF)));
    Wd0 = digital_update_groupconnected(Wa0,H);
    Wd0 = sqrt(Pt)/max(norm(Wa0*Wd0,'fro'),1e-12)*Wd0;
    W0 = Wa0*Wd0;

    [obj0,sr0,crlb0,FIM0] = ...
        evaluate_metrics(Pre,H,W0,delta_c,delta_s,noise_c,K);

    %% Conventional SSP: N50 prefix + long reference
    W = W0; Wa = Wa0; Wd = Wd0; FIM = FIM0;
    obj_cur = obj0; sr_cur = sr0; crlb_cur = crlb0;

    obj_B1_50 = obj0; sr_B1_50 = sr0; crlb_B1_50 = crlb0;
    ref_iter = 0;

    for t = 1:reference_iter

        W_old = W; Wa_old = Wa; Wd_old = Wd;
        obj_old = obj_cur; sr_old = sr_cur; crlb_old = crlb_cur;

        [grad_total,lambda_local] = ...
            compute_grad_and_local_curvature( ...
                Pre,H,W_old,FIM,Nbasis,R_N, ...
                delta_c,delta_s,noise_c,K,M,mu_floor,mu_margin);

        lambda_trial = lambda_local;
        accepted = false;

        for trial = 1:max_accept_trials

            W_target = grad_total + lambda_trial*W_old;

            [Wa_cand,Wd_cand] = ...
                matrix_approximation_groupconnected_fixed( ...
                    W_target,Wa_old,Wd_old,Z, ...
                    hbf_outer_iter,hbf_inner_iter);

            Wd_cand = sqrt(Pt)/max(norm(Wa_cand*Wd_cand,'fro'),1e-12)*Wd_cand;
            W_cand = Wa_cand*Wd_cand;

            [obj_cand,sr_cand,crlb_cand,FIM_cand] = ...
                evaluate_metrics(Pre,H,W_cand,delta_c,delta_s,noise_c,K);

            if obj_cand >= obj_old-b1_accept_tol
                accepted = true;
                W = W_cand; Wa = Wa_cand; Wd = Wd_cand; FIM = FIM_cand;
                obj_cur = obj_cand; sr_cur = sr_cand; crlb_cur = crlb_cand;
                break;
            end

            lambda_trial = rho_in*lambda_trial;
        end

        if ~accepted
            W = W_old; Wa = Wa_old; Wd = Wd_old;
            obj_cur = obj_old; sr_cur = sr_old; crlb_cur = crlb_old;
            FIM = calculateFIM(Pre,W_old);
        end

        if t==Nexe
            obj_B1_50 = obj_cur;
            sr_B1_50 = sr_cur;
            crlb_B1_50 = crlb_cur;
        end

        ref_iter = t;

        relchg = abs([obj_cur,crlb_cur,sr_cur]-[obj_old,crlb_old,sr_old]) ./ ...
            max(abs([obj_old,crlb_old,sr_old]),1e-12);

        if t>=Nexe && all(relchg<=tol)
            break;
        end
    end

    if ref_iter<Nexe
        obj_B1_50 = obj_cur;
        sr_B1_50 = sr_cur;
        crlb_B1_50 = crlb_cur;
    end

    obj_ref = obj_cur;
    sr_ref = sr_cur;
    crlb_ref = crlb_cur;

    %% Frozen Deep schedule
    [Obj_traj,SR_traj,CRLB_traj,Reject_cum] = ...
        run_b3_schedule( ...
            Pre,H,Nbasis,R_N,Z,W0,Wa0,Wd0,FIM0, ...
            gamma_B3,beta_B3,stage1_layers,Ndeep, ...
            delta_c,delta_s,noise_c,K,M,Pt, ...
            hbf_outer_iter,hbf_inner_iter, ...
            mu_floor,mu_margin,deep_accept_tol);

    obj_Deep50 = Obj_traj(Nexe+1);
    sr_Deep50 = SR_traj(Nexe+1);
    crlb_Deep50 = CRLB_traj(Nexe+1);

    obj_Deep200 = Obj_traj(Ndeep+1);
    sr_Deep200 = SR_traj(Ndeep+1);
    crlb_Deep200 = CRLB_traj(Ndeep+1);

    reject_Deep50 = Reject_cum(Nexe+1);
    reject_Deep200 = Reject_cum(Ndeep+1);

    [gap_B1_50,gap_Deep50,gap_Deep200] = ...
        compute_three_gaps(obj_ref,obj0,obj_B1_50,obj_Deep50,obj_Deep200);

    out = struct();
    out.obj0 = obj0; out.sr0 = sr0; out.crlb0 = crlb0;
    out.obj_ref = obj_ref; out.sr_ref = sr_ref; out.crlb_ref = crlb_ref;
    out.ref_iter = ref_iter;

    out.obj_B1_50 = obj_B1_50;
    out.sr_B1_50 = sr_B1_50;
    out.crlb_B1_50 = crlb_B1_50;

    out.obj_Deep50 = obj_Deep50;
    out.sr_Deep50 = sr_Deep50;
    out.crlb_Deep50 = crlb_Deep50;

    out.obj_Deep200 = obj_Deep200;
    out.sr_Deep200 = sr_Deep200;
    out.crlb_Deep200 = crlb_Deep200;

    out.gap_B1_50 = gap_B1_50;
    out.gap_Deep50 = gap_Deep50;
    out.gap_Deep200 = gap_Deep200;

    out.reject_Deep50 = reject_Deep50;
    out.reject_Deep200 = reject_Deep200;
end

function [Obj_traj,SR_traj,CRLB_traj,Reject_cum] = ...
    run_b3_schedule( ...
        Pre,H,Nbasis,R_N,Z,W0,Wa0,Wd0,FIM0, ...
        gamma,beta,stage1_layers,Ndeep, ...
        delta_c,delta_s,noise_c,K,M,Pt, ...
        hbf_outer_iter,hbf_inner_iter, ...
        mu_floor,mu_margin,accept_tol)

    W = W0; Wa = Wa0; Wd = Wd0; FIM = FIM0;
    [obj_cur,sr_cur,crlb_cur,~] = ...
        evaluate_metrics(Pre,H,W,delta_c,delta_s,noise_c,K);

    Obj_traj = nan(1,Ndeep+1);
    SR_traj = nan(1,Ndeep+1);
    CRLB_traj = nan(1,Ndeep+1);
    Reject_cum = zeros(1,Ndeep+1);

    Obj_traj(1)=obj_cur; SR_traj(1)=sr_cur; CRLB_traj(1)=crlb_cur;

    W_prev = W0;
    reject_count = 0;

    for t = 1:Ndeep

        W_old = W; Wa_old = Wa; Wd_old = Wd;
        obj_old = obj_cur; sr_old = sr_cur; crlb_old = crlb_cur;

        [grad_total,lambda_local] = ...
            compute_grad_and_local_curvature( ...
                Pre,H,W_old,FIM,Nbasis,R_N, ...
                delta_c,delta_s,noise_c,K,M,mu_floor,mu_margin);

        gamma_t = gamma(t);

        if t<=stage1_layers
            beta_t = 0;
        else
            beta_t = beta(t);
        end

        inertial_state = W_old + beta_t*(W_old-W_prev);
        effective_shift = lambda_local/max(gamma_t,1e-8);

        W_target = grad_total + effective_shift*inertial_state;

        [Wa_cand,Wd_cand] = ...
            matrix_approximation_groupconnected_fixed( ...
                W_target,Wa_old,Wd_old,Z,hbf_outer_iter,hbf_inner_iter);

        Wd_cand = sqrt(Pt)/max(norm(Wa_cand*Wd_cand,'fro'),1e-12)*Wd_cand;
        W_cand = Wa_cand*Wd_cand;

        [obj_cand,sr_cand,crlb_cand,FIM_cand] = ...
            evaluate_metrics(Pre,H,W_cand,delta_c,delta_s,noise_c,K);

        if obj_cand >= obj_old-accept_tol
            W_new = W_cand; Wa_new = Wa_cand; Wd_new = Wd_cand;
            FIM_new = FIM_cand;
            obj_new = obj_cand; sr_new = sr_cand; crlb_new = crlb_cand;
        else
            W_new = W_old; Wa_new = Wa_old; Wd_new = Wd_old;
            FIM_new = FIM;
            obj_new = obj_old; sr_new = sr_old; crlb_new = crlb_old;
            reject_count = reject_count+1;
        end

        W_prev = W_old;
        W = W_new; Wa = Wa_new; Wd = Wd_new; FIM = FIM_new;
        obj_cur = obj_new; sr_cur = sr_new; crlb_cur = crlb_new;

        Obj_traj(t+1)=obj_cur;
        SR_traj(t+1)=sr_cur;
        CRLB_traj(t+1)=crlb_cur;
        Reject_cum(t+1)=reject_count;
    end
end

function S = init_sweep_storage(ncfg,nch)

    fields = { ...
        'Obj0','SR0','CRLB0', ...
        'ObjRef','SRRef','CRLBRef','RefIter', ...
        'Obj_B1_50','SR_B1_50','CRLB_B1_50', ...
        'Obj_Deep50','SR_Deep50','CRLB_Deep50', ...
        'Obj_Deep200','SR_Deep200','CRLB_Deep200', ...
        'Gap_B1_50','Gap_Deep50','Gap_Deep200', ...
        'Reject_Deep50','Reject_Deep200'};

    S = struct();
    for k = 1:numel(fields)
        S.(fields{k}) = nan(ncfg,nch);
    end
end

function S = store_case_result(S,i,c,out)

    names = fieldnames(S);

    map = struct( ...
        'Obj0','obj0','SR0','sr0','CRLB0','crlb0', ...
        'ObjRef','obj_ref','SRRef','sr_ref','CRLBRef','crlb_ref', ...
        'RefIter','ref_iter', ...
        'Obj_B1_50','obj_B1_50','SR_B1_50','sr_B1_50','CRLB_B1_50','crlb_B1_50', ...
        'Obj_Deep50','obj_Deep50','SR_Deep50','sr_Deep50','CRLB_Deep50','crlb_Deep50', ...
        'Obj_Deep200','obj_Deep200','SR_Deep200','sr_Deep200','CRLB_Deep200','crlb_Deep200', ...
        'Gap_B1_50','gap_B1_50','Gap_Deep50','gap_Deep50','Gap_Deep200','gap_Deep200', ...
        'Reject_Deep50','reject_Deep50','Reject_Deep200','reject_Deep200');

    for k = 1:numel(names)
        S.(names{k})(i,c) = out.(map.(names{k}));
    end
end

function S = finalize_sweep_storage(S)

    fields = fieldnames(S);

    for k = 1:numel(fields)

        name = fields{k};
        X = S.(name);

        if ~isnumeric(X) || ndims(X)~=2
            continue;
        end

        S.([name '_avg']) = mean(X,2,'omitnan');

        nvalid = sum(isfinite(X),2);
        S.([name '_CI95']) = ...
            1.96*std(X,0,2,'omitnan') ./ sqrt(max(nvalid,1));
    end

    gain = S.Gap_B1_50-S.Gap_Deep50;
    S.Gain_Deep50_vs_B1_50 = gain;
    S.Gain_Deep50_vs_B1_50_avg = mean(gain,2,'omitnan');

    nvalid = sum(isfinite(gain),2);
    S.Gain_Deep50_vs_B1_50_CI95 = ...
        1.96*std(gain,0,2,'omitnan') ./ sqrt(max(nvalid,1));
end

function save_checkpoint(save_file, ...
    Pt_dBm_list,Pt_list,Nt_list,Nth_list,Ntv_list, ...
    Ptres,Ntres,done_Pt,done_Nt,H_Nt_nested, ...
    K,M,Nr,N_RF,G,Pt0_dBm,Pt0,noise_c,noise_s,L,rho0, ...
    delta_c0,delta_s0,Nexe,Ndeep,stage1_layers, ...
    gamma_B3,beta_B3)

    save(save_file, ...
        'Pt_dBm_list','Pt_list','Nt_list','Nth_list','Ntv_list', ...
        'Ptres','Ntres','done_Pt','done_Nt','H_Nt_nested', ...
        'K','M','Nr','N_RF','G','Pt0_dBm','Pt0', ...
        'noise_c','noise_s','L','rho0','delta_c0','delta_s0', ...
        'Nexe','Ndeep','stage1_layers','gamma_B3','beta_B3', ...
        '-v7.3');
end

function [grad_total,lambda_local] = ...
    compute_grad_and_local_curvature( ...
        Pre,H,W,FIM,Nbasis,R_N,delta_c,delta_s, ...
        noise_c,K,M,mu_floor,mu_margin)

    T_k = sum(square_abs(H'*W),2)+noise_c*ones(K,1);

    alpha_k = ...
        T_k ./ (T_k-square_abs(diag(H'*W))) - 1;

    beta_k = ...
        sqrt(1+alpha_k).*diag(H'*W)./T_k;

    Sigma1 = diag(sqrt(1+alpha_k).*beta_k);
    Sigma2 = diag(square_abs(beta_k));

    CRBM = inv(FIM);
    Phi = CRBM*CRBM;

    Sigma3 = construct_Sigma3(Pre,Phi);

    Omega1 = [delta_c*Sigma1; zeros(3*M,K)];
    Omega2 = blkdiag(-delta_c*Sigma2,delta_s*Sigma3);

    [lambda_exact,~] = ...
        compute_lowdim_exact_shift_from_Omega2(R_N,Omega2);

    lambda_local = max(lambda_exact+mu_margin,mu_floor);

    grad_total = Nbasis*(Omega1+Omega2*(Nbasis'*W));
end

function [obj,sr,crlb,FIM] = ...
    evaluate_metrics(Pre,H,W,delta_c,delta_s,noise_c,K)

    FIM = calculateFIM(Pre,W);
    crlb = trace(inv(FIM));

    T_k = sum(square_abs(H'*W),2)+noise_c*ones(K,1);

    rate_p = log(T_k ./ ...
        (T_k-square_abs(diag(H'*W))));

    sr = sum(rate_p);
    obj = delta_c*sr-delta_s*crlb;
end

function [g1,g2,g3] = compute_three_gaps(fref,f0,f1,f2,f3)

    denom = fref-f0;
    epsden = 1e-12*max(1,max(abs([fref,f0])));

    if abs(denom)<=epsden
        g1=0; g2=0; g3=0;
        return;
    end

    g1 = max(fref-f1,0)/(denom+epsden);
    g2 = max(fref-f2,0)/(denom+epsden);
    g3 = max(fref-f3,0)/(denom+epsden);
end

function [mu_exact,eig_red] = ...
    compute_lowdim_exact_shift_from_Omega2(R_N,Omega2)

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
        U=Q; R_basis=R; r=0; return;
    end

    d = abs(diag(R));

    if isempty(d)
        r=0;
    else
        tol_rank = max(size(X))*eps(class(X))*max(d);
        r = sum(d>tol_rank);
    end

    U = Q(:,1:r);
    R_basis = R(1:r,:);
end

function Sigma3 = construct_Sigma3(Pre,Phi)

    M = size(Pre.tilde_B_oo,1);

    phi11=Phi(1:M,1:M);
    phi12=Phi(1:M,M+1:2*M);
    phi13=Phi(1:M,2*M+1:3*M);
    phi14=Phi(1:M,3*M+1:4*M);

    phi22=Phi(M+1:2*M,M+1:2*M);
    phi23=Phi(M+1:2*M,2*M+1:3*M);
    phi24=Phi(M+1:2*M,3*M+1:4*M);

    phi33=Phi(2*M+1:3*M,2*M+1:3*M);
    phi34=Phi(2*M+1:3*M,3*M+1:4*M);
    phi44=Phi(3*M+1:4*M,3*M+1:4*M);

    Boo=Pre.tilde_B_oo;
    Both=Pre.tilde_B_oth;
    Boph=Pre.tilde_B_oph;
    Btho=Pre.tilde_B_tho;
    Bthth=Pre.tilde_B_thth;
    Bthph=Pre.tilde_B_thph;
    Bpho=Pre.tilde_B_pho;
    Bphph=Pre.tilde_B_phph;

    Moo=phi11.*Bthth+2*phi12.*Bthph+phi22.*Bphph;
    Mtho=2*(phi11.*Both+phi12.*Boph);
    Mpho=2*(phi22.*Boph);
    Moph=2*(phi12.*Btho);
    Mthph=2*(phi12.*Boo);
    Mthth=phi11.*Boo;
    Mphph=phi22.*Boo;

    Mo3=(2*phi13+2j*phi14).*Btho + (2*phi23+2j*phi24).*Bpho;
    Mth3=(2*phi13+2j*phi14).*Boo;
    Mph3=(2*phi23+2j*phi24).*Boo;
    M33=(phi33+phi44+2j*phi34).*Boo;

    Gamma=Pre.Gamma;

    idxA=1:M;
    idxTh=M+1:2*M;
    idxPh=2*M+1:3*M;

    Sigma3_tilde=zeros(3*M,3*M);

    Sigma3_tilde(idxA,idxA)=Sigma3_tilde(idxA,idxA)+Gamma'*Moo*Gamma;
    Sigma3_tilde(idxTh,idxA)=Sigma3_tilde(idxTh,idxA)+Gamma'*Mtho*Gamma;
    Sigma3_tilde(idxPh,idxA)=Sigma3_tilde(idxPh,idxA)+Gamma'*Mpho*Gamma;
    Sigma3_tilde(idxA,idxPh)=Sigma3_tilde(idxA,idxPh)+Gamma'*Moph*Gamma;
    Sigma3_tilde(idxTh,idxPh)=Sigma3_tilde(idxTh,idxPh)+Gamma'*Mthph*Gamma;
    Sigma3_tilde(idxPh,idxPh)=Sigma3_tilde(idxPh,idxPh)+Gamma'*Mphph*Gamma;
    Sigma3_tilde(idxTh,idxTh)=Sigma3_tilde(idxTh,idxTh)+Gamma'*Mthth*Gamma;
    Sigma3_tilde(idxA,idxA)=Sigma3_tilde(idxA,idxA)+Gamma'*Mo3;
    Sigma3_tilde(idxTh,idxA)=Sigma3_tilde(idxTh,idxA)+Gamma'*Mth3;
    Sigma3_tilde(idxPh,idxA)=Sigma3_tilde(idxPh,idxA)+Gamma'*Mph3;
    Sigma3_tilde(idxA,idxA)=Sigma3_tilde(idxA,idxA)+M33;

    Sigma3=(2*Pre.L/Pre.noise_s)*(Sigma3_tilde+Sigma3_tilde')/2;
end

function FIM = calculateFIM(Pre,D)

    Wo=Pre.tilde_A_o*D;
    Wth=Pre.tilde_A_th*D;
    Wph=Pre.tilde_A_ph*D;
    Wa=Pre.tilde_A*D;

    Goo=Wo*Wo';
    Goth=Wo*Wth';
    Gtho=Wth*Wo';
    Goph=Wo*Wph';
    Gpho=Wph*Wo';
    Gthth=Wth*Wth';
    Gphph=Wph*Wph';
    Gath=Wa*Wth';
    Gao=Wa*Wo';
    Gaph=Wa*Wph';
    Gaa=Wa*Wa';
    Gphth=Wph*Wth';

    F11=(Goo).'.*Pre.tilde_B_thth + ...
        (Goth).'.*Pre.tilde_B_oth + ...
        (Gtho).'.*Pre.tilde_B_tho + ...
        (Gthth).'.*Pre.tilde_B_oo;

    F12=(Goo).'.*Pre.tilde_B_thph + ...
        (Goth).'.*Pre.tilde_B_oph + ...
        (Gpho).'.*Pre.tilde_B_tho + ...
        (Gphth).'.*Pre.tilde_B_oo;

    F13=(Gao).'.*Pre.tilde_B_tho + ...
        (Gath).'.*Pre.tilde_B_oo;

    F22=(Goo).'.*Pre.tilde_B_phph + ...
        (Goph).'.*Pre.tilde_B_oph + ...
        (Gpho).'.*Pre.tilde_B_pho + ...
        (Gphph).'.*Pre.tilde_B_oo;

    F23=(Gao).'.*Pre.tilde_B_pho + ...
        (Gaph).'.*Pre.tilde_B_oo;

    F33=(Gaa).'.*Pre.tilde_B_oo;

    M=size(Pre.tilde_B_oo,1);

    FIM=zeros(4*M);

    FIM(1:M,1:M)=real(F11);
    FIM(1:M,M+1:2*M)=real(F12);
    FIM(1:M,2*M+1:3*M)=real(F13);
    FIM(1:M,3*M+1:4*M)=-imag(F13);

    FIM(M+1:2*M,M+1:2*M)=real(F22);
    FIM(M+1:2*M,2*M+1:3*M)=real(F23);
    FIM(M+1:2*M,3*M+1:4*M)=-imag(F23);

    FIM(2*M+1:3*M,2*M+1:3*M)=real(F33);
    FIM(2*M+1:3*M,3*M+1:4*M)=-imag(F33);
    FIM(3*M+1:4*M,3*M+1:4*M)=real(F33);

    FIM=triu(FIM)+triu(FIM,1)';
    FIM=2*Pre.L/Pre.noise_s*FIM;
end

function [A,dAtheta,dAphi] = ...
    construct_steer_matrix_and_derivative_steer_matrix(theta,phi,Mx,My)

    ix=(0:Mx-1)';
    iy=(0:My-1)';

    M=length(theta);

    A=zeros(Mx*My,M);
    dAtheta=zeros(Mx*My,M);
    dAphi=zeros(Mx*My,M);

    for m=1:M

        ax=1/sqrt(Mx)*exp(1j*pi*ix*sin(theta(m))*sin(phi(m)));
        ay=1/sqrt(My)*exp(1j*pi*iy*cos(phi(m)));

        daxtheta=1j*pi*ix*cos(theta(m))*sin(phi(m)).*ax;
        daxphi=1j*pi*ix*sin(theta(m))*cos(phi(m)).*ax;
        dayphi=-1j*pi*iy*sin(phi(m)).*ay;

        A(:,m)=kron(ay,ax);
        dAtheta(:,m)=kron(ay,daxtheta);
        dAphi(:,m)=kron(ay,daxphi)+kron(dayphi,ax);
    end
end

function [Wa,Wd] = ...
    matrix_approximation_groupconnected_fixed( ...
        Wtarget,Wa,Wd,Z,outer_iter,inner_iter)

    Wa=Z.*exp(1j*angle(Wa));

    for count=1:outer_iter

        Wd=digital_update_groupconnected(Wa,Wtarget);

        Rwd=Wd*Wd';
        Rwd=(Rwd+Rwd')/2;

        mu1=gershgorin_shift(Rwd);

        for ii=1:inner_iter

            D=-Wa*Rwd + Wtarget*Wd' + mu1*Wa;
            Wa=Z.*exp(1j*angle(D));
        end
    end
end

function mu=gershgorin_shift(C)

    diag_abs=abs(diag(C));
    row_sum=sum(abs(C),2)-diag_abs;
    diag_real=real(diag(C));

    mu=max(diag_real+row_sum)+1e-6;
end

function Wd=digital_update_groupconnected(Wa,Wtarget)
    Wd=Wa\Wtarget;
end

function Z=build_groupconnected_mask(Nt,N_RF,G)

    if mod(Nt,G)~=0 || mod(N_RF,G)~=0
        error('G must divide Nt and N_RF.');
    end

    Nant_g=Nt/G;
    Nrf_g=N_RF/G;

    Z=zeros(Nt,N_RF);

    for gg=1:G
        ant_idx=(gg-1)*Nant_g+(1:Nant_g);
        rf_idx=(gg-1)*Nrf_g+(1:Nrf_g);
        Z(ant_idx,rf_idx)=1;
    end
end

function y=db2power(x)
    y=10.^(x/10);
end

function y=square_abs(x)
    y=abs(x).^2;
end
