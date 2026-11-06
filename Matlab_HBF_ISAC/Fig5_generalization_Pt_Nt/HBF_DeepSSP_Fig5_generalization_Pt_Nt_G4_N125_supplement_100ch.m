%% 
clc
clear
close all

%% ========================================================================
% Fig. 5 SUPPLEMENT: Deep-SSP N=125 generalization over Pt and Nt
%
% This lightweight supplement computes ONLY the missing frozen Deep-SSP
% results at the selected practical depth N = 125.
%
% It reuses the already-finished Fig. 5 simulation for:
%   * the exact Pt grid,
%   * the exact Nt grid / UPA dimensions,
%   * the exact nested Nt-sweep channel realization H_Nt_nested,
%   * the exact shared nominal Pt-sweep dataset,
%   * the same-channel initial/reference objective values,
%   * SSP N=50, Deep N=50, Deep N=200, and long-run SSP reference.
%
% Therefore NO conventional SSP or long-run reference is recomputed.
%
% Output:
% Fig5_generalization_Pt_Nt/results/
%   data_Fig5_DeepSSP_generalization_Pt_Nt_G4_N125_supplement_100ch.mat
%% ========================================================================

%% Paths
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

base_file = fullfile(fig_dir,'results', ...
    'data_Fig5_DeepSSP_generalization_Pt_Nt_G4_100ch.mat');

if ~exist(dataset_file,'file')
    error('Cannot find fixed dataset:\n%s',dataset_file);
end

if ~exist(param_file,'file')
    error('Cannot find trained parameter file:\n%s',param_file);
end

if ~exist(base_file,'file')
    error(['Cannot find the completed Pt/Nt base simulation:\n%s\n' ...
           'Run the original Fig. 5 Pt/Nt simulation first.'],base_file);
end

Dset = load(dataset_file);
Plearn = load(param_file);
Base = load(base_file);

%% Required fields
required_base = { ...
    'Pt_dBm_list','Pt_list', ...
    'Nt_list','Nth_list','Ntv_list', ...
    'Ptres','Ntres','H_Nt_nested', ...
    'K','M','Nr','N_RF','G', ...
    'Pt0_dBm','Pt0','noise_c','noise_s','L','rho0', ...
    'delta_c0','delta_s0'};

for k = 1:numel(required_base)
    if ~isfield(Base,required_base{k})
        error('Base result missing variable "%s".',required_base{k});
    end
end

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

%% Shared configuration
Pt_dBm_list = double(Base.Pt_dBm_list(:).');
Pt_list = double(Base.Pt_list(:).');

Nt_list = double(Base.Nt_list(:).');
Nth_list = double(Base.Nth_list(:).');
Ntv_list = double(Base.Ntv_list(:).');

Ptres_base = Base.Ptres;
Ntres_base = Base.Ntres;

H_Nt_nested = Base.H_Nt_nested;

K = double(Base.K);
M = double(Base.M);

Nr = double(Base.Nr);
N_RF = double(Base.N_RF);
G = double(Base.G);

Pt0_dBm = double(Base.Pt0_dBm);
Pt0 = double(Base.Pt0);

noise_c = double(Base.noise_c);
noise_s = double(Base.noise_s);
L = double(Base.L);

rho0 = double(Base.rho0);
delta_c0 = double(Base.delta_c0);
delta_s0 = double(Base.delta_s0);

Nrh = 10;
Nrv = 8;

H_test_all = Dset.H_test_all;
alpha_test = Dset.alpha_test;
theta_test = Dset.theta_test;
phi_test   = Dset.phi_test;

I_out = size(Ptres_base.ObjRef,2);

%% Frozen G=4 parent schedule, execute to N=125 only
G_learn = double(Plearn.G_list(:).');
idxG = find(G_learn==G,1,'first');

if isempty(idxG)
    error('G=%d not found in trained parameter file.',G);
end

Nexe = 125;
stage1_layers = double(Plearn.stage1_layers);

gamma_all = double(Plearn.gamma_B3);
beta_all = double(Plearn.beta_B3);

if size(gamma_all,2)<Nexe || size(beta_all,2)<Nexe
    error('Learned schedule contains fewer than %d layers.',Nexe);
end

gamma_B3 = gamma_all(idxG,1:Nexe);
beta_B3 = beta_all(idxG,1:Nexe);

%% Projection / safeguard settings
hbf_outer_iter = 5;
hbf_inner_iter = 2;

deep_accept_tol = 1e-8;

mu_floor = 1e-8;
mu_margin = 1e-8;

%% Storage
Pt125 = init_supplement_storage(numel(Pt_list),I_out);
Nt125 = init_supplement_storage(numel(Nt_list),I_out);

done_Pt = false(numel(Pt_list),I_out);
done_Nt = false(numel(Nt_list),I_out);

result_dir = fullfile(fig_dir,'results');
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

save_file = fullfile(result_dir, ...
    'data_Fig5_DeepSSP_generalization_Pt_Nt_G4_N125_supplement_100ch.mat');

%% Resume
resume_existing = true;

if resume_existing && exist(save_file,'file')

    R = load(save_file);

    compatible = ...
        isfield(R,'Pt_dBm_list') && isfield(R,'Nt_list') && ...
        isfield(R,'Nexe') && isfield(R,'G') && ...
        isequal(double(R.Pt_dBm_list(:).'),Pt_dBm_list) && ...
        isequal(double(R.Nt_list(:).'),Nt_list) && ...
        double(R.Nexe)==Nexe && double(R.G)==G && ...
        isfield(R,'Pt125') && isfield(R,'Nt125') && ...
        isfield(R,'done_Pt') && isfield(R,'done_Nt');

    if compatible
        Pt125 = R.Pt125;
        Nt125 = R.Nt125;
        done_Pt = R.done_Pt;
        done_Nt = R.done_Nt;

        fprintf('Resume accepted: Pt %d/%d, Nt %d/%d cases complete.\n', ...
            nnz(done_Pt),numel(done_Pt),nnz(done_Nt),numel(done_Nt));
    else
        warning('Existing N=125 checkpoint is incompatible; starting fresh.');
    end
end

fprintf('============================================================\n');
fprintf('Fig. 5 N=125 Pt/Nt generalization supplement\n');
fprintf('G=%d, selected Nexe=%d\n',G,Nexe);
fprintf('Pt sweep = [%s] dBm\n',num2str(Pt_dBm_list));
fprintf('Nt sweep = [%s]\n',num2str(Nt_list));
fprintf('Channels/configuration = %d\n',I_out);
fprintf('NO conventional SSP/reference recomputation.\n');
fprintf('============================================================\n');

%% ========================================================================
% Pt sweep: exact nominal fixed Nt=64 dataset
%% ========================================================================
fprintf('\n================ Pt SWEEP: N=125 only ================\n');

Nth = 8;
Ntv = 8;
Nt_nom = 64;

Z = build_groupconnected_mask(Nt_nom,N_RF,G);

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

        out = run_deep_only_case( ...
            H,H_seed,alpha,theta,phi, ...
            Nth,Ntv,Nrh,Nrv,Z,Pt,noise_c,noise_s,L, ...
            delta_c0,delta_s0, ...
            gamma_B3,beta_B3,stage1_layers,Nexe, ...
            hbf_outer_iter,hbf_inner_iter, ...
            mu_floor,mu_margin,deep_accept_tol);

        fref = Ptres_base.ObjRef(iP,channel);
        f0 = Ptres_base.Obj0(iP,channel);

        out.gap = normalized_gap(fref,f0,out.obj);

        Pt125.Obj(iP,channel) = out.obj;
        Pt125.SR(iP,channel) = out.sr;
        Pt125.CRLB(iP,channel) = out.crlb;
        Pt125.Gap(iP,channel) = out.gap;
        Pt125.Reject(iP,channel) = out.reject;

        done_Pt(iP,channel) = true;

        if channel==1 || mod(channel,10)==0 || channel==I_out
            fprintf(['Pt=%g dBm, channel %3d/%3d | ' ...
                     'Deep125 gap=%.5f | reject=%d\n'], ...
                Pt_dBm,channel,I_out,out.gap,out.reject);
        end

        if mod(channel,5)==0 || channel==I_out
            save_supplement_checkpoint(save_file, ...
                Pt_dBm_list,Pt_list,Nt_list,Nth_list,Ntv_list, ...
                Pt125,Nt125,done_Pt,done_Nt,Nexe,G,K,M,Nr,N_RF, ...
                Pt0_dBm,Pt0,noise_c,noise_s,L,rho0, ...
                delta_c0,delta_s0,stage1_layers,gamma_B3,beta_B3);
        end
    end
end

%% ========================================================================
% Nt sweep: reuse EXACT nested channels from completed base simulation
%% ========================================================================
fprintf('\n================ Nt SWEEP: N=125 only ================\n');

for iN = 1:numel(Nt_list)

    Nt = Nt_list(iN);
    Nth = Nth_list(iN);
    Ntv = Ntv_list(iN);

    Z = build_groupconnected_mask(Nt,N_RF,G);

    fprintf('\n---- Nt = %d (%dx%d), %d/%d ----\n', ...
        Nt,Nth,Ntv,iN,numel(Nt_list));

    for channel = 1:I_out

        if done_Nt(iN,channel)
            continue;
        end

        H = double(squeeze(H_Nt_nested(channel,1:Nt,1:K)));
        H_seed = H;

        alpha = double(alpha_test(channel,1:M).');
        theta = double(theta_test(channel,1:M).');
        phi   = double(phi_test(channel,1:M).');

        out = run_deep_only_case( ...
            H,H_seed,alpha,theta,phi, ...
            Nth,Ntv,Nrh,Nrv,Z,Pt0,noise_c,noise_s,L, ...
            delta_c0,delta_s0, ...
            gamma_B3,beta_B3,stage1_layers,Nexe, ...
            hbf_outer_iter,hbf_inner_iter, ...
            mu_floor,mu_margin,deep_accept_tol);

        fref = Ntres_base.ObjRef(iN,channel);
        f0 = Ntres_base.Obj0(iN,channel);

        out.gap = normalized_gap(fref,f0,out.obj);

        Nt125.Obj(iN,channel) = out.obj;
        Nt125.SR(iN,channel) = out.sr;
        Nt125.CRLB(iN,channel) = out.crlb;
        Nt125.Gap(iN,channel) = out.gap;
        Nt125.Reject(iN,channel) = out.reject;

        done_Nt(iN,channel) = true;

        if channel==1 || mod(channel,10)==0 || channel==I_out
            fprintf(['Nt=%d, channel %3d/%3d | ' ...
                     'Deep125 gap=%.5f | reject=%d\n'], ...
                Nt,channel,I_out,out.gap,out.reject);
        end

        if mod(channel,5)==0 || channel==I_out
            save_supplement_checkpoint(save_file, ...
                Pt_dBm_list,Pt_list,Nt_list,Nth_list,Ntv_list, ...
                Pt125,Nt125,done_Pt,done_Nt,Nexe,G,K,M,Nr,N_RF, ...
                Pt0_dBm,Pt0,noise_c,noise_s,L,rho0, ...
                delta_c0,delta_s0,stage1_layers,gamma_B3,beta_B3);
        end
    end
end

%% Final averages
Pt125 = finalize_supplement_storage(Pt125);
Nt125 = finalize_supplement_storage(Nt125);

save_supplement_checkpoint(save_file, ...
    Pt_dBm_list,Pt_list,Nt_list,Nth_list,Ntv_list, ...
    Pt125,Nt125,done_Pt,done_Nt,Nexe,G,K,M,Nr,N_RF, ...
    Pt0_dBm,Pt0,noise_c,noise_s,L,rho0, ...
    delta_c0,delta_s0,stage1_layers,gamma_B3,beta_B3);

fprintf('\n============================================================\n');
fprintf('N=125 Pt/Nt supplement complete.\n');
fprintf('Saved to:\n%s\n',save_file);

fprintf('\nPt sweep Deep125 gap:\n');
fprintf('Pt(dBm) | gap125 | rejects125\n');
for iP = 1:numel(Pt_dBm_list)
    fprintf('%7.1f | %.5f | %.2f\n', ...
        Pt_dBm_list(iP),Pt125.Gap_avg(iP),Pt125.Reject_avg(iP));
end

fprintf('\nNt sweep Deep125 gap:\n');
fprintf(' Nt | gap125 | rejects125\n');
for iN = 1:numel(Nt_list)
    fprintf('%3d | %.5f | %.2f\n', ...
        Nt_list(iN),Nt125.Gap_avg(iN),Nt125.Reject_avg(iN));
end

fprintf('============================================================\n');

%% ========================================================================
% Supplement-specific helpers
%% ========================================================================
function S = init_supplement_storage(ncfg,nch)

    S = struct();

    S.Obj = nan(ncfg,nch);
    S.SR = nan(ncfg,nch);
    S.CRLB = nan(ncfg,nch);
    S.Gap = nan(ncfg,nch);
    S.Reject = nan(ncfg,nch);
end


function S = finalize_supplement_storage(S)

    names = {'Obj','SR','CRLB','Gap','Reject'};

    for k = 1:numel(names)

        X = S.(names{k});

        S.([names{k} '_avg']) = mean(X,2,'omitnan');

        nvalid = sum(isfinite(X),2);

        S.([names{k} '_CI95']) = ...
            1.96*std(X,0,2,'omitnan') ./ sqrt(max(nvalid,1));
    end
end


function save_supplement_checkpoint(save_file, ...
    Pt_dBm_list,Pt_list,Nt_list,Nth_list,Ntv_list, ...
    Pt125,Nt125,done_Pt,done_Nt,Nexe,G,K,M,Nr,N_RF, ...
    Pt0_dBm,Pt0,noise_c,noise_s,L,rho0, ...
    delta_c0,delta_s0,stage1_layers,gamma_B3,beta_B3)

    save(save_file, ...
        'Pt_dBm_list','Pt_list','Nt_list','Nth_list','Ntv_list', ...
        'Pt125','Nt125','done_Pt','done_Nt', ...
        'Nexe','G','K','M','Nr','N_RF','Pt0_dBm','Pt0', ...
        'noise_c','noise_s','L','rho0','delta_c0','delta_s0', ...
        'stage1_layers','gamma_B3','beta_B3','-v7.3');
end


function out = run_deep_only_case( ...
    H,H_seed,alpha,theta,phi, ...
    Nth,Ntv,Nrh,Nrv,Z,Pt,noise_c,noise_s,L, ...
    delta_c,delta_s, ...
    gamma,beta,stage1_layers,Nexe, ...
    hbf_outer_iter,hbf_inner_iter, ...
    mu_floor,mu_margin,accept_tol)

    K = size(H,2);
    M = numel(theta);
    N_RF = size(Z,2);

    [A,dAtheta,dAphi] = ...
        construct_steer_matrix_and_derivative_steer_matrix( ...
            theta,phi,Nth,Ntv);

    [B,dBtheta,dBphi] = ...
        construct_steer_matrix_and_derivative_steer_matrix( ...
            theta,phi,Nrh,Nrv);

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

    Wa0 = Z.*exp(1j*angle(H_seed(:,1:N_RF)));

    Wd0 = digital_update_groupconnected(Wa0,H);
    Wd0 = sqrt(Pt)/max(norm(Wa0*Wd0,'fro'),1e-12)*Wd0;

    W0 = Wa0*Wd0;

    [~,~,~,FIM0] = ...
        evaluate_metrics( ...
            Pre,H,W0,delta_c,delta_s,noise_c,K);

    [Obj_traj,SR_traj,CRLB_traj,Reject_cum] = ...
        run_b3_schedule( ...
            Pre,H,Nbasis,R_N,Z, ...
            W0,Wa0,Wd0,FIM0, ...
            gamma,beta,stage1_layers,Nexe, ...
            delta_c,delta_s,noise_c,K,M,Pt, ...
            hbf_outer_iter,hbf_inner_iter, ...
            mu_floor,mu_margin,accept_tol);

    out = struct();

    out.obj = Obj_traj(end);
    out.sr = SR_traj(end);
    out.crlb = CRLB_traj(end);
    out.reject = Reject_cum(end);
end


function g = normalized_gap(fref,f0,f)

    denom = fref-f0;

    epsden = ...
        1e-12*max(1,max(abs([fref,f0])));

    if abs(denom)<=epsden
        g = 0;
    else
        g = max(fref-f,0)/(denom+epsden);
    end
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
