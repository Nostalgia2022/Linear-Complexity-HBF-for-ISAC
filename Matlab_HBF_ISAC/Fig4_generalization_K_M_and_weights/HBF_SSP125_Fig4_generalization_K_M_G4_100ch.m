clc
clear
close all

%% ========================================================================
% Fig. 4 SUPPLEMENT: equal-iteration SSP-HBF baseline, N = 125
%
% Purpose
% -------
% Generate the conventional SSP-HBF result after exactly N = 125 updates
% for the same K/M generalization cases used in the existing Fig. 4 data.
% This provides a direct equal-iteration comparison against
% Deep-SSP-HBF with N_exe = 125.
%
% IMPORTANT
% ---------
% * Same 100 fixed communication-channel realizations.
% * Same M-target realizations saved in the original Fig. 4 result file.
% * Same G = 4 architecture, initialization, objective weights, and HBF
%   projection settings.
% * SSP-HBF is executed for exactly 125 iterations; no convergence-based
%   early stopping is used in this supplement.
% * This file computes ONLY the missing SSP-HBF N=125 baseline.
%
% Output:
% Fig4_generalization_K_M/results/
%   data_Fig4_SSP125_generalization_K_M_G4_100ch.mat
%% ========================================================================

%% -------------------- Paths ---------------------------------------------
script_fullpath = mfilename('fullpath');

if isempty(script_fullpath)
    fig_dir = pwd;
else
    fig_dir = fileparts(script_fullpath);
end

project_root = fileparts(fig_dir);

dataset_file = fullfile(project_root, ...
    'fixed_dataset_K8_Nt64_M3.mat');

base_file = fullfile(fig_dir,'results', ...
    'data_Fig4_DeepSSP_generalization_K_M_G4_100ch.mat');

if ~exist(dataset_file,'file')
    error('Cannot find fixed dataset:\n%s',dataset_file);
end

if ~exist(base_file,'file')
    error('Cannot find original Fig. 4 result file:\n%s',base_file);
end

Dset = load(dataset_file);
B0 = load(base_file);

%% -------------------- Validate original experiment ----------------------
required_base = { ...
    'K_list','M_list','M_targets', ...
    'K0','M0','Nt','Nr','N_RF','G','Z', ...
    'Pt','noise_c','noise_s','L', ...
    'delta_c0','delta_s0','Kres','Mres'};

for k = 1:numel(required_base)
    if ~isfield(B0,required_base{k})
        error('Original Fig. 4 result missing "%s".',required_base{k});
    end
end

required_data = {'H_test_all','alpha_test','theta_test','phi_test'};

for k = 1:numel(required_data)
    if ~isfield(Dset,required_data{k})
        error('Fixed dataset missing "%s".',required_data{k});
    end
end

H_test_all = Dset.H_test_all;
alpha_test = Dset.alpha_test;
theta_test = Dset.theta_test;
phi_test   = Dset.phi_test;

K_list = double(B0.K_list(:).');
M_list = double(B0.M_list(:).');

K0 = double(B0.K0);
M0 = double(B0.M0);

Nt = double(B0.Nt);
Nr = double(B0.Nr);
N_RF = double(B0.N_RF);
G = double(B0.G);
Z = double(B0.Z);

Pt = double(B0.Pt);
noise_c = double(B0.noise_c);
noise_s = double(B0.noise_s);
L = double(B0.L);

delta_c = double(B0.delta_c0);
delta_s = double(B0.delta_s0);

M_targets = B0.M_targets;

% Array dimensions used by the original nominal experiment.
Nth = 8;
Ntv = 8;
Nrh = 10;
Nrv = 8;

if Nth*Ntv ~= Nt || Nrh*Nrv ~= Nr
    error('Stored Nt/Nr are inconsistent with the nominal UPA dimensions.');
end

I_out = size(B0.Kres.SRRef,2);

%% -------------------- Equal iteration budget ----------------------------
N_ssp = 125;

% Same SSP / HBF settings as the original Fig. 4 generator.
hbf_outer_iter = 5;
hbf_inner_iter = 2;

max_accept_trials = 20;
rho_in = 2;

b1_accept_tol = 1e-10;

mu_floor = 1e-8;
mu_margin = 1e-8;

fprintf('============================================================\n');
fprintf('Fig. 4 equal-iteration SSP-HBF supplement\n');
fprintf('SSP iteration budget : N = %d\n',N_ssp);
fprintf('G = %d, K0 = %d, M0 = %d, channels = %d\n', ...
    G,K0,M0,I_out);
fprintf('K sweep = [%s]\n',num2str(K_list));
fprintf('M sweep = [%s]\n',num2str(M_list));
fprintf('============================================================\n');

%% -------------------- Storage / checkpoint ------------------------------
KSSP125 = init_storage(numel(K_list),I_out);
MSSP125 = init_storage(numel(M_list),I_out);

done_K = false(numel(K_list),I_out);
done_M = false(numel(M_list),I_out);

result_dir = fullfile(fig_dir,'results');

if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

save_file = fullfile(result_dir, ...
    'data_Fig4_SSP125_generalization_K_M_G4_100ch.mat');

resume_existing = true;

if resume_existing && exist(save_file,'file')

    R = load(save_file);

    compatible = ...
        isfield(R,'K_list') && isfield(R,'M_list') && ...
        isfield(R,'N_ssp') && isfield(R,'G') && ...
        isequal(double(R.K_list(:).'),K_list) && ...
        isequal(double(R.M_list(:).'),M_list) && ...
        double(R.N_ssp)==N_ssp && double(R.G)==G && ...
        isfield(R,'KSSP125') && isfield(R,'MSSP125') && ...
        isfield(R,'done_K') && isfield(R,'done_M');

    if compatible
        KSSP125 = R.KSSP125;
        MSSP125 = R.MSSP125;
        done_K = R.done_K;
        done_M = R.done_M;

        fprintf('Resume accepted: K %d/%d, M %d/%d cases complete.\n\n', ...
            nnz(done_K),numel(done_K),nnz(done_M),numel(done_M));
    else
        warning('Existing SSP125 checkpoint incompatible; starting fresh.');
    end
end

%% ========================================================================
% K sweep
%% ========================================================================
fprintf('\n================ K SWEEP: SSP125 ================\n');

for iK = 1:numel(K_list)

    Kc = K_list(iK);

    fprintf('\n---- K = %d (%d/%d) ----\n',Kc,iK,numel(K_list));

    for channel = 1:I_out

        if done_K(iK,channel)
            continue;
        end

        H_seed = double(squeeze(H_test_all(channel,:,1:K0)));
        H = H_seed(:,1:Kc);

        alpha = double(alpha_test(channel,1:M0).');
        theta = double(theta_test(channel,1:M0).');
        phi   = double(phi_test(channel,1:M0).');

        out = run_ssp_fixed_depth( ...
            H,H_seed,alpha,theta,phi, ...
            Nth,Ntv,Nrh,Nrv,Z, ...
            Pt,noise_c,noise_s,L, ...
            delta_c,delta_s,N_ssp, ...
            hbf_outer_iter,hbf_inner_iter, ...
            max_accept_trials,rho_in,b1_accept_tol, ...
            mu_floor,mu_margin);

        KSSP125 = store_result(KSSP125,iK,channel,out);
        done_K(iK,channel) = true;

        if channel==1 || mod(channel,10)==0 || channel==I_out
            fprintf(['K=%d, channel %3d/%3d | SR=%.5f | ' ...
                     'CRLB=%.6e | obj=%+.6e\n'], ...
                Kc,channel,I_out,out.sr,out.crlb,out.obj);
        end

        if mod(channel,5)==0 || channel==I_out
            save_ssp125_checkpoint( ...
                save_file,K_list,M_list,KSSP125,MSSP125, ...
                done_K,done_M,N_ssp,G,K0,M0, ...
                Pt,noise_c,noise_s,L,delta_c,delta_s);
        end
    end
end

%% ========================================================================
% M sweep -- EXACTLY reuse M_targets from the original Fig. 4 file
%% ========================================================================
fprintf('\n================ M SWEEP: SSP125 ================\n');

for iM = 1:numel(M_list)

    Mc = M_list(iM);
    target_set = M_targets{iM};

    fprintf('\n---- M = %d (%d/%d) ----\n',Mc,iM,numel(M_list));

    for channel = 1:I_out

        if done_M(iM,channel)
            continue;
        end

        H_seed = double(squeeze(H_test_all(channel,:,1:K0)));
        H = H_seed;

        alpha = double(target_set.alpha(channel,:).');
        theta = double(target_set.theta(channel,:).');
        phi   = double(target_set.phi(channel,:).');

        out = run_ssp_fixed_depth( ...
            H,H_seed,alpha,theta,phi, ...
            Nth,Ntv,Nrh,Nrv,Z, ...
            Pt,noise_c,noise_s,L, ...
            delta_c,delta_s,N_ssp, ...
            hbf_outer_iter,hbf_inner_iter, ...
            max_accept_trials,rho_in,b1_accept_tol, ...
            mu_floor,mu_margin);

        MSSP125 = store_result(MSSP125,iM,channel,out);
        done_M(iM,channel) = true;

        if channel==1 || mod(channel,10)==0 || channel==I_out
            fprintf(['M=%d, channel %3d/%3d | SR=%.5f | ' ...
                     'CRLB=%.6e | obj=%+.6e\n'], ...
                Mc,channel,I_out,out.sr,out.crlb,out.obj);
        end

        if mod(channel,5)==0 || channel==I_out
            save_ssp125_checkpoint( ...
                save_file,K_list,M_list,KSSP125,MSSP125, ...
                done_K,done_M,N_ssp,G,K0,M0, ...
                Pt,noise_c,noise_s,L,delta_c,delta_s);
        end
    end
end

%% -------------------- Final averages ------------------------------------
KSSP125 = finalize_storage(KSSP125);
MSSP125 = finalize_storage(MSSP125);

save_ssp125_checkpoint( ...
    save_file,K_list,M_list,KSSP125,MSSP125, ...
    done_K,done_M,N_ssp,G,K0,M0, ...
    Pt,noise_c,noise_s,L,delta_c,delta_s);

fprintf('\n============================================================\n');
fprintf('SSP-HBF N=125 supplement complete.\n');
fprintf('Saved to:\n%s\n',save_file);

fprintf('\nK sweep:\n');
fprintf(' K | SR SSP125 | CRLB SSP125 | time (s)\n');
for i = 1:numel(K_list)
    fprintf('%2d | %9.5f | %11.5e | %8.4f\n', ...
        K_list(i), ...
        KSSP125.SR_avg(i),KSSP125.CRLB_avg(i),KSSP125.Time_avg(i));
end

fprintf('\nM sweep:\n');
fprintf(' M | SR SSP125 | CRLB SSP125 | time (s)\n');
for i = 1:numel(M_list)
    fprintf('%2d | %9.5f | %11.5e | %8.4f\n', ...
        M_list(i), ...
        MSSP125.SR_avg(i),MSSP125.CRLB_avg(i),MSSP125.Time_avg(i));
end
fprintf('============================================================\n');

%% ========================================================================
% One case: conventional SSP-HBF with a FIXED N-iteration budget
%% ========================================================================
function out = run_ssp_fixed_depth( ...
    H,H_seed,alpha,theta,phi, ...
    Nth,Ntv,Nrh,Nrv,Z, ...
    Pt,noise_c,noise_s,L, ...
    delta_c,delta_s,N_ssp, ...
    hbf_outer_iter,hbf_inner_iter, ...
    max_accept_trials,rho_in,b1_accept_tol, ...
    mu_floor,mu_margin)

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

    if size(H_seed,2)<N_RF
        error('H_seed must contain at least N_RF=%d columns.',N_RF);
    end

    % Same deterministic GC-feasible initialization used by Fig. 4.
    Wa = Z.*exp(1j*angle(H_seed(:,1:N_RF)));

    Wd = digital_update_groupconnected(Wa,H);
    Wd = sqrt(Pt)/max(norm(Wa*Wd,'fro'),1e-12)*Wd;

    W = Wa*Wd;

    [obj_cur,sr_cur,crlb_cur,FIM] = ...
        evaluate_metrics(Pre,H,W,delta_c,delta_s,noise_c,K);

    accepted_count = 0;
    backtrack_count = 0;

    tstart = tic;

    for t = 1:N_ssp

        W_old = W;
        Wa_old = Wa;
        Wd_old = Wd;

        obj_old = obj_cur;
        sr_old = sr_cur;
        crlb_old = crlb_cur;
        FIM_old = FIM;

        [grad_total,lambda_local] = ...
            compute_grad_and_local_curvature( ...
                Pre,H,W_old,FIM_old,Nbasis,R_N, ...
                delta_c,delta_s,noise_c,K,M, ...
                mu_floor,mu_margin);

        accepted = false;
        lambda_trial = lambda_local;

        for trial = 1:max_accept_trials

            W_target = grad_total+lambda_trial*W_old;

            [Wa_cand,Wd_cand] = ...
                matrix_approximation_groupconnected_fixed( ...
                    W_target,Wa_old,Wd_old,Z, ...
                    hbf_outer_iter,hbf_inner_iter);

            Wd_cand = ...
                sqrt(Pt)/max(norm(Wa_cand*Wd_cand,'fro'),1e-12)*Wd_cand;

            W_cand = Wa_cand*Wd_cand;

            [obj_cand,sr_cand,crlb_cand,FIM_cand] = ...
                evaluate_metrics( ...
                    Pre,H,W_cand,delta_c,delta_s,noise_c,K);

            if obj_cand >= obj_old-b1_accept_tol

                accepted = true;

                W = W_cand;
                Wa = Wa_cand;
                Wd = Wd_cand;
                FIM = FIM_cand;

                obj_cur = obj_cand;
                sr_cur = sr_cand;
                crlb_cur = crlb_cand;

                accepted_count = accepted_count+1;
                backtrack_count = backtrack_count+(trial-1);

                break;
            end

            lambda_trial = rho_in*lambda_trial;
        end

        if ~accepted
            % Retain the last accepted iterate, but continue so the method
            % is evaluated under the same fixed 125-iteration budget.
            W = W_old;
            Wa = Wa_old;
            Wd = Wd_old;
            FIM = FIM_old;

            obj_cur = obj_old;
            sr_cur = sr_old;
            crlb_cur = crlb_old;

            backtrack_count = backtrack_count+max_accept_trials;
        end
    end

    elapsed = toc(tstart);

    out = struct();
    out.obj = obj_cur;
    out.sr = sr_cur;
    out.crlb = crlb_cur;
    out.time = elapsed;
    out.accepted = accepted_count;
    out.backtracks = backtrack_count;
end


function S = init_storage(ncfg,nch)

    fields = {'Obj','SR','CRLB','Time','Accepted','Backtracks'};

    S = struct();

    for k = 1:numel(fields)
        S.(fields{k}) = nan(ncfg,nch);
    end
end


function S = store_result(S,i,c,out)

    S.Obj(i,c) = out.obj;
    S.SR(i,c) = out.sr;
    S.CRLB(i,c) = out.crlb;
    S.Time(i,c) = out.time;
    S.Accepted(i,c) = out.accepted;
    S.Backtracks(i,c) = out.backtracks;
end


function S = finalize_storage(S)

    names = fieldnames(S);

    for k = 1:numel(names)

        name = names{k};

        if endsWith(name,'_avg') || endsWith(name,'_CI95')
            continue;
        end

        X = S.(name);

        if ~isnumeric(X) || ndims(X)~=2
            continue;
        end

        S.([name '_avg']) = mean(X,2,'omitnan');

        nvalid = sum(isfinite(X),2);
        S.([name '_CI95']) = ...
            1.96*std(X,0,2,'omitnan') ./ sqrt(max(nvalid,1));
    end
end


function save_ssp125_checkpoint( ...
    save_file,K_list,M_list,KSSP125,MSSP125,done_K,done_M, ...
    N_ssp,G,K0,M0,Pt,noise_c,noise_s,L,delta_c,delta_s)

    save(save_file, ...
        'K_list','M_list','KSSP125','MSSP125','done_K','done_M', ...
        'N_ssp','G','K0','M0','Pt','noise_c','noise_s','L', ...
        'delta_c','delta_s','-v7.3');
end


function [grad_total,lambda_local] = ...
    compute_grad_and_local_curvature( ...
        Pre,H,W,FIM,Nbasis,R_N,delta_c,delta_s, ...
        noise_c,K,M,mu_floor,mu_margin)

    T_k = ...
        sum(square_abs(H'*W),2) + ...
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
        (Omega1+Omega2*(Nbasis'*W));
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


function [g1,g2,g3] = ...
    compute_three_gaps(fref,f0,f1,f2,f3)

    denom = fref-f0;

    epsden = ...
        1e-12*max(1,max(abs([fref,f0])));

    if abs(denom) <= epsden
        g1 = 0;
        g2 = 0;
        g3 = 0;
        return;
    end

    g1 = max(fref-f1,0)/(denom+epsden);
    g2 = max(fref-f2,0)/(denom+epsden);
    g3 = max(fref-f3,0)/(denom+epsden);
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

    d = abs(diag(R));

    if isempty(d)
        r = 0;
    else
        tol_rank = ...
            max(size(X))*eps(class(X))*max(d);
        r = sum(d>tol_rank);
    end

    U = Q(:,1:r);
    R_basis = R(1:r,:);
end

%% ========================================================================
% Sensing expressions
%% ========================================================================
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

    Mtho = ...
        2*(phi11.*Both+phi12.*Boph);

    Mpho = ...
        2*(phi22.*Boph);

    Moph = ...
        2*(phi12.*Btho);

    Mthph = ...
        2*(phi12.*Boo);

    Mthth = ...
        phi11.*Boo;

    Mphph = ...
        phi22.*Boo;

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
        Sigma3_tilde(idxA,idxA) + ...
        M33;

    Sigma3 = ...
        (2*Pre.L/Pre.noise_s) * ...
        (Sigma3_tilde+Sigma3_tilde')/2;
end


function FIM = calculateFIM(Pre,D)

    Wo  = Pre.tilde_A_o*D;
    Wth = Pre.tilde_A_th*D;
    Wph = Pre.tilde_A_ph*D;
    Wa  = Pre.tilde_A*D;

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

    FIM = triu(FIM)+triu(FIM,1)';

    FIM = ...
        2*Pre.L/Pre.noise_s*FIM;
end

%% ========================================================================
% Array response
%% ========================================================================
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

%% ========================================================================
% Fixed-budget group-connected HBF projection
%% ========================================================================
function [Wa,Wd] = ...
    matrix_approximation_groupconnected_fixed( ...
        Wtarget,Wa,Wd,Z,outer_iter,inner_iter)

    Wa = Z.*exp(1j*angle(Wa));

    for count = 1:outer_iter

        Wd = ...
            digital_update_groupconnected( ...
                Wa,Wtarget);

        Rwd = Wd*Wd';
        Rwd = (Rwd+Rwd')/2;

        % Gershgorin inner surrogate shift matching the final implementation.
        mu1 = gershgorin_shift(Rwd);

        for ii = 1:inner_iter

            D = ...
                -Wa*Rwd + ...
                Wtarget*Wd' + ...
                mu1*Wa;

            Wa = ...
                Z.*exp(1j*angle(D));
        end
    end
end


function mu = gershgorin_shift(C)

    diag_abs = abs(diag(C));

    row_sum = ...
        sum(abs(C),2)-diag_abs;

    diag_real = ...
        real(diag(C));

    mu = ...
        max(diag_real+row_sum)+1e-6;
end


function Wd = ...
    digital_update_groupconnected( ...
        Wa,Wtarget)

    Wd = Wa\Wtarget;
end


function Z = ...
    build_groupconnected_mask( ...
        Nt,N_RF,G)

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
    Nrf_g = N_RF/G;

    Z = zeros(Nt,N_RF);

    for gg = 1:G

        ant_idx = ...
            (gg-1)*Nant_g + (1:Nant_g);

        rf_idx = ...
            (gg-1)*Nrf_g + (1:Nrf_g);

        Z(ant_idx,rf_idx) = 1;
    end
end

%% ========================================================================
% Small utilities
%% ========================================================================
function y = db2power(x)
    y = 10.^(x/10);
end


function y = square_abs(x)
    y = abs(x).^2;
end
