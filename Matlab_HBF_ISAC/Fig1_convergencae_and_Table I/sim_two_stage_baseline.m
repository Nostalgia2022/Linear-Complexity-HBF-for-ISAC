clc
clear
close all

%% ========================================================================
% Two-stage FD -> HBF baseline for the CONVERGENCE FIGURE (G = 4, nominal)
%
% Same 100 test channels, same weight (kappa = 1e-3), same initialization
% W0 = E0 P0 as SSP-HBF / Deep-SSP-HBF in the convergence figure.
%   Stage 1: fully digital SSP from W0 (closed-form power projection, same
%            gradient/shift/backtracking/stopping rule as SSP-HBF)
%            -> stationary fully digital W_FD
%   Stage 2: ONE decomposition of W_FD by Algorithm 1 (I_AO=200, I_in=20),
%            analog phases initialized from W_FD, P rescaled to full power.
%
% Saved per channel: f0, objective/SR/CRLB of W_FD and of the two-stage HBF,
% fitting error, FD iterations, stage-1 and stage-2 CPU times.
% If the convergence-figure data file
%   data_TableI_v2_full_vs_52_N125_run200_100ch.mat
% is found, the normalized objective progress of the two-stage baseline is
% computed with the SAME reference (SSP-HBF full after 2000 iterations) and
% the SAME channel average (mean of per-channel ratios) as the figure.
%% ========================================================================

%% -------------------- Options --------------------
G = 4;
lambda_nom = 1e3;          % delta_s/delta_c, i.e. kappa = 1e-3
n_dec_init = 1;            % 1 = standard single AltMin from the phases of W_FD
I_out = 100;

%% -------------------- Paths and data --------------------
script_fullpath = mfilename('fullpath');
if isempty(script_fullpath), fig_dir = pwd; else, fig_dir = fileparts(script_fullpath); end
project_root = fileparts(fig_dir);

dataset_file = first_existing({ ...
    fullfile(project_root,'fixed_dataset_K8_Nt64_M3.mat'), ...
    fullfile(fig_dir,'fixed_dataset_K8_Nt64_M3.mat')});
Dset = load(dataset_file);
H_test_all = Dset.H_test_all;
alpha_test = Dset.alpha_test;
theta_test = Dset.theta_test;
phi_test   = Dset.phi_test;

%% -------------------- System setting (as in the paper) --------------------
M = 3;  K = 8;
Nth = 8;  Ntv = 8;  Nt = Nth*Ntv;
Nrh = 10; Nrv = 8;
N_RF = K;
Pt      = db2power(20);
noise_c = db2power(0);
noise_s = db2power(0);
L = 64;
delta_c = 1/(1+lambda_nom);
delta_s = lambda_nom/(1+lambda_nom);

Pssp.tol = 1e-5;
Pssp.count_max = 2000;
Pssp.max_accept_trials = 20;
Pssp.eps_dec = 1e-10;
Pssp.rho_in = 2;
Pssp.mu_floor = 1e-8;
Pssp.mu_margin = 1e-8;
Pssp.use_stopping_test = true;

Z = build_groupconnected_mask(Nt,N_RF,G);

out_dir = fullfile(fig_dir,'results');
if ~exist(out_dir,'dir'), mkdir(out_dir); end
save_file = fullfile(out_dir,sprintf('data_TwoStage_FD_HBF_G%d_nominal_100ch.mat',G));

pool = gcp('nocreate');
if isempty(pool), pool = parpool; end

%% -------------------- Simulation --------------------
f0 = nan(I_out,1);
FD_obj = nan(I_out,1); FD_sr = nan(I_out,1); FD_crlb = nan(I_out,1);
FD_iter = nan(I_out,1); FD_time = nan(I_out,1);
TS_obj = nan(I_out,1); TS_sr = nan(I_out,1); TS_crlb = nan(I_out,1);
TS_fit = nan(I_out,1); TS_aoit = nan(I_out,1); TS_dectime = nan(I_out,1);

tAll = tic;
parfor ch = 1:I_out
    H = double(squeeze(H_test_all(ch,:,1:K)));
    alpha = double(alpha_test(ch,1:M).');
    theta = double(theta_test(ch,1:M).');
    phi   = double(phi_test(ch,1:M).');

    [A,dAtheta,dAphi] = construct_steer_matrix_and_derivative_steer_matrix(theta,phi,Nth,Ntv);
    [B,dBtheta,dBphi] = construct_steer_matrix_and_derivative_steer_matrix(theta,phi,Nrh,Nrv);
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
    Pre.L = L; Pre.noise_s = noise_s; Pre.gamma = alpha; Pre.Gamma = Gamma;

    Nbasis = [H,A,dAtheta,dAphi];
    [~,R_N,~] = qr_rank_basis(Nbasis);

    % identical initialization to the convergence figure
    Wa0 = Z .* exp(1j*angle(H(:,1:N_RF)));
    Wd0 = digital_update_groupconnected(Wa0,H(:,1:K));
    Wd0 = sqrt(Pt)/max(norm(Wa0*Wd0,'fro'),1e-12)*Wd0;
    W0 = Wa0*Wd0;
    [o0,s0,c0,F0] = evaluate_metrics(Pre,H,W0,delta_c,delta_s,noise_c,K);
    f0(ch) = o0;

    rng(1000+ch,'twister');   % only used if n_dec_init > 1

    % ---------- stage 1: stationary fully digital beamformer ----------
    R1 = run_ssp_fd(Pre,H,W0,F0,o0,s0,c0,Nbasis,R_N,Pt,delta_c,delta_s,noise_c,K,M,Pssp);
    FD_obj(ch) = R1.obj; FD_sr(ch) = R1.sr; FD_crlb(ch) = R1.crlb;
    FD_iter(ch) = R1.iter; FD_time(ch) = R1.time;

    % ---------- stage 2: one-shot hybrid decomposition ----------
    t2 = tic;
    [Wa_ts,Wd_ts,err_ts,ao_ts] = decompose_once(R1.W,Z,N_RF,n_dec_init);
    Wd_ts = sqrt(Pt)/max(norm(Wa_ts*Wd_ts,'fro'),1e-12)*Wd_ts;
    [o2,s2,c2] = evaluate_metrics(Pre,H,Wa_ts*Wd_ts,delta_c,delta_s,noise_c,K);
    TS_dectime(ch) = toc(t2);
    TS_obj(ch) = o2; TS_sr(ch) = s2; TS_crlb(ch) = c2;
    TS_fit(ch) = err_ts; TS_aoit(ch) = ao_ts;
end
fprintf('Finished in %.1f s wall time.\n',toc(tAll));

TS_time = FD_time + TS_dectime;     % total two-stage CPU time per channel

save(save_file,'G','lambda_nom','n_dec_init','Pssp','I_out','f0', ...
    'FD_obj','FD_sr','FD_crlb','FD_iter','FD_time', ...
    'TS_obj','TS_sr','TS_crlb','TS_fit','TS_aoit','TS_dectime','TS_time','-v7.3');

%% -------------------- Summary --------------------
fprintf('\n===== Two-stage FD -> HBF, G = %d, kappa = %g =====\n',G,1/lambda_nom);
fprintf('Fully digital (stage 1): SR %.3f, CRLB %.4e, %.0f iterations, %.3f s\n', ...
    mean(FD_sr),mean(FD_crlb),mean(FD_iter),mean(FD_time));
fprintf('Two-stage HBF (stage 2): SR %.3f, CRLB %.4e (median %.4e)\n', ...
    mean(TS_sr),mean(TS_crlb),median(TS_crlb));
fprintf('  fitting error ||W_FD-EP||/||W_FD|| : mean %.3f, max %.3f\n',mean(TS_fit),max(TS_fit));
fprintf('  total CPU time (stage 1 + 2)       : %.3f s (decomposition %.3f s)\n', ...
    mean(TS_time),mean(TS_dectime));
fprintf('  objective below initialization on %d/%d channels\n',nnz(TS_obj < f0),I_out);

% ---------- normalized progress with the convergence-figure reference ----------
fig_name = 'data_TableI_v2_full_vs_52_N125_run200_100ch.mat';
cand = {fullfile(fig_dir,'results',fig_name),fullfile(fig_dir,fig_name), ...
        fullfile(project_root,'results',fig_name)};
ref_file = '';
for k = 1:numel(cand), if exist(cand{k},'file'), ref_file = cand{k}; break; end, end
if isempty(ref_file)
    hit = dir(fullfile(project_root,'**',fig_name));
    if ~isempty(hit), ref_file = fullfile(hit(1).folder,hit(1).name); end
end
if isempty(ref_file)
    warning('Convergence-figure data not found, normalized progress not computed.');
else
    Dref = load(ref_file);
    r = Dref.Res([Dref.Res.G]==G);
    fprintf('\nReference file: %s\n',ref_file);
    fprintf('  max |f0 - f0_fig| = %.3e (should be ~0, same initialization)\n', ...
        max(abs(f0 - r.f0(:))));
    dd = max(r.full.obj(:) - r.f0(:),1e-12);
    prog_ts = (TS_obj - r.f0(:))./dd;
    prog_fd = (FD_obj - r.f0(:))./dd;
    fprintf('  normalized progress, two-stage HBF : %.4f (MoR), median %.4f, min %.4f\n', ...
        mean(prog_ts),median(prog_ts),min(prog_ts));
    fprintf('  normalized progress, fully digital : %.4f (upper bound, not HBF-feasible)\n', ...
        mean(prog_fd));
    fprintf('  SSP-HBF (full) reference          : SR %.3f, CRLB %.4e\n', ...
        mean(r.full.sr),mean(r.full.crlb));
    save(save_file,'prog_ts','prog_fd','ref_file','-append');
end

%% ========================================================================
% Local functions
% ========================================================================
function R = run_ssp_fd(Pre,H,W,FIM,obj0,sr0,crlb0,Nbasis,R_N,Pt, ...
    delta_c,delta_s,noise_c,K,M,P)
%RUN_SSP_FD Fully digital SSP: same gradient, shift, backtracking and
% stopping rule as SSP-HBF, but the shifted beamformer is projected onto
% {||W||_F^2 = Pt} in closed form (no hybrid decomposition).
    Con_prev = [obj0, crlb0, sr0];
    Con_new = Con_prev;
    last_iter = 1;
    t0 = tic;
    for iter = 2:P.count_max
        W_prev = W; obj_prev = Con_prev(1);
        [grad_total,mu] = compute_grad_and_local_curvature( ...
            Pre,H,W_prev,FIM,Nbasis,R_N,delta_c,delta_s,noise_c,K,M, ...
            P.mu_floor,P.mu_margin);
        accepted = false; trial = 0;
        while ~accepted && trial < P.max_accept_trials
            trial = trial + 1;
            W_f = grad_total + mu*W_prev;
            W_c = sqrt(Pt)*W_f/max(norm(W_f,'fro'),1e-12);
            [o_c,s_c,c_c,F_c] = evaluate_metrics(Pre,H,W_c,delta_c,delta_s,noise_c,K);
            if o_c >= obj_prev - P.eps_dec
                accepted = true;
                W = W_c; FIM = F_c; Con_new = [o_c, c_c, s_c]; last_iter = iter;
            else
                mu = P.rho_in*mu;
            end
        end
        if ~accepted, break; end
        relchg = abs(Con_new-Con_prev)./max(abs(Con_prev),1e-12);
        Con_prev = Con_new;
        if P.use_stopping_test && all(relchg <= P.tol), break; end
    end
    R.W = W; R.obj = Con_prev(1); R.crlb = Con_prev(2); R.sr = Con_prev(3);
    R.iter = last_iter - 1; R.time = toc(t0);
end


function [Wa,Wd,rel_err,ao_iter] = decompose_once(W_fd,Z,N_RF,n_init)
%DECOMPOSE_ONCE One-shot hybrid decomposition of a fully digital
% beamformer by the alternating decomposition of Algorithm 1 (full budget).
% Init 1: analog phases from W_FD (columns 1..N_RF, padded if N_RF > K).
% Inits 2..n_init: random phases. Keeps the smallest fitting error.
    K = size(W_fd,2);
    best = inf;
    for k = 1:n_init
        if k == 1
            if N_RF <= K
                X = W_fd(:,1:N_RF);
            else
                X = [W_fd, exp(1j*2*pi*rand(size(W_fd,1),N_RF-K))];
            end
            Wa_init = Z .* exp(1j*angle(X));
        else
            Wa_init = Z .* exp(1j*2*pi*rand(size(Z)));
        end
        [~,Wa_k,~,it_k] = matrix_approximation_groupconnected(W_fd,Wa_init,[],Z);
        Wd_k = digital_update_groupconnected(Wa_k,W_fd);   % LS for the final E
        e = norm(W_fd - Wa_k*Wd_k,'fro')/max(norm(W_fd,'fro'),1e-12);
        if e < best
            best = e; Wa = Wa_k; Wd = Wd_k; ao_iter = it_k;
        end
    end
    rel_err = best;
end


function f = first_existing(c)
    for k = 1:numel(c)
        if exist(c{k},'file'), f = c{k}; return; end
    end
    error('None of these files exist:\n%s',strjoin(c,newline));
end


%% ========================================================================
% Helper functions copied unchanged from
%   HBF_DeepSSP_Fig2_tradeoff_GC_N200_diagnostic_100ch.m
% ========================================================================


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


%% ========================================================================
% Helper functions copied unchanged from
%   HBF_SGPI_tradeoff_GC_parfor_checkpoint_100ch.m
% ========================================================================


function [mu_exact, eig_red] = compute_lowdim_exact_shift_from_Omega2(R_N, Omega2)
%COMPUTE_LOWDIM_EXACT_SHIFT_FROM_OMEGA2 Exact reduced-dimensional shift.
%
% For the compact quadratic coefficient
%   C2 = Nbasis*Omega2*Nbasis',  Nbasis = U_N*R_N,
% the nonzero eigenvalues of C2 are the eigenvalues of
%   C2_red = R_N*Omega2*R_N'.
% This avoids applying C2 to U_N and avoids forming any Nt-by-Nt matrix.

    C2_red = R_N * Omega2 * R_N';

    % Remove tiny numerical non-Hermitian errors.
    C2_red = (C2_red + C2_red') / 2;

    eig_red = real(eig(C2_red));

    if isempty(eig_red)
        mu_exact = 0;
    else
        mu_exact = max(abs(eig_red));
    end
end


function [U, R_basis, r] = qr_rank_basis(X)
%QR_RANK_BASIS Orthonormal basis of range(X) using economy QR.
%
% X = U*R_basis after truncating numerical rank.  U has orthonormal
% columns, and R_basis keeps all original columns of X in the reduced basis.
    [Q, R] = qr(X, 0);
    if isempty(R)
        U = Q;
        R_basis = R;
        r = 0;
        return;
    end
    tol_rank = max(size(X)) * eps(class(X)) * max(abs(diag(R)));
    r = sum(abs(diag(R)) > tol_rank);
    U = Q(:, 1:r);
    R_basis = R(1:r, :);
end


function Sigma3 = construct_Sigma3(Pre, Phi)
%CONSTRUCT_SIGMA3 Construct the exact small sensing coefficient.
%
% This function returns Sigma3 such that
%   compute_QW(Pre,Phi,W) = S*Sigma3*(S'*W),
% where S=[A,dAtheta,dAphi].  The construction follows the same M-domain
% matrices used in compute_QW.  First, a non-Hermitian Sigma3_tilde is
% assembled for Q_tilde. Then
%   Sigma3 = (2L/noise_s)*(Sigma3_tilde+Sigma3_tilde')/2.

    M = size(Pre.tilde_B_oo, 1);

    % Partition Phi
    phi11 = Phi(1:M, 1:M);
    phi12 = Phi(1:M, M+1:2*M);
    phi13 = Phi(1:M, 2*M+1:3*M);
    phi14 = Phi(1:M, 3*M+1:4*M);

    phi22 = Phi(M+1:2*M, M+1:2*M);
    phi23 = Phi(M+1:2*M, 2*M+1:3*M);
    phi24 = Phi(M+1:2*M, 3*M+1:4*M);

    phi33 = Phi(2*M+1:3*M, 2*M+1:3*M);
    phi34 = Phi(2*M+1:3*M, 3*M+1:4*M);
    phi44 = Phi(3*M+1:4*M, 3*M+1:4*M);

    % Short names for B-domain matrices
    Boo   = Pre.tilde_B_oo;
    Both  = Pre.tilde_B_oth;
    Boph  = Pre.tilde_B_oph;
    Btho  = Pre.tilde_B_tho;
    Bthth = Pre.tilde_B_thth;
    Bthph = Pre.tilde_B_thph;
    Bpho  = Pre.tilde_B_pho;
    Bphph = Pre.tilde_B_phph;

    % Same M-domain coefficient matrices as in compute_QW
    Moo   = phi11 .* Bthth + 2*phi12 .* Bthph + phi22 .* Bphph;
    Mtho  = 2*(phi11 .* Both + phi12 .* Boph);
    Mpho  = 2*(phi22 .* Boph);
    Moph  = 2*(phi12 .* Btho);
    Mthph = 2*(phi12 .* Boo);

    Mthth  = phi11 .* Boo;
    Mphph  = phi22 .* Boo;

    Mo3   = (2*phi13 + 2j*phi14) .* Btho + (2*phi23 + 2j*phi24) .* Bpho;
    Mth3  = (2*phi13 + 2j*phi14) .* Boo;
    Mph3  = (2*phi23 + 2j*phi24) .* Boo;
    M33   = (phi33 + phi44 + 2j*phi34) .* Boo;

    Gamma = Pre.Gamma;

    % Sigma3_tilde is arranged according to S=[A,dAtheta,dAphi].
    % A term S_i*T*S_j' contributes T to block (i,j).
    idxA  = 1:M;
    idxTh = M+1:2*M;
    idxPh = 2*M+1:3*M;

    Sigma3_tilde = zeros(3*M, 3*M);

    % Terms originally involving Ato = Gamma*A', Atth = Gamma*dAtheta',
    % and Atph = Gamma*dAphi'. The Gamma factors are absorbed into the
    % corresponding coefficient blocks.
    Sigma3_tilde(idxA,  idxA)  = Sigma3_tilde(idxA,  idxA)  + Gamma' * Moo   * Gamma;
    Sigma3_tilde(idxTh, idxA)  = Sigma3_tilde(idxTh, idxA)  + Gamma' * Mtho  * Gamma;
    Sigma3_tilde(idxPh, idxA)  = Sigma3_tilde(idxPh, idxA)  + Gamma' * Mpho  * Gamma;

    Sigma3_tilde(idxA,  idxPh) = Sigma3_tilde(idxA,  idxPh) + Gamma' * Moph  * Gamma;
    Sigma3_tilde(idxTh, idxPh) = Sigma3_tilde(idxTh, idxPh) + Gamma' * Mthph * Gamma;
    Sigma3_tilde(idxPh, idxPh) = Sigma3_tilde(idxPh, idxPh) + Gamma' * Mphph * Gamma;

    Sigma3_tilde(idxTh, idxTh) = Sigma3_tilde(idxTh, idxTh) + Gamma' * Mthth * Gamma;

    % Terms with right-side At=A' do not have the right Gamma factor.
    Sigma3_tilde(idxA,  idxA)  = Sigma3_tilde(idxA,  idxA)  + Gamma' * Mo3;
    Sigma3_tilde(idxTh, idxA)  = Sigma3_tilde(idxTh, idxA)  + Gamma' * Mth3;
    Sigma3_tilde(idxPh, idxA)  = Sigma3_tilde(idxPh, idxA)  + Gamma' * Mph3;
    Sigma3_tilde(idxA,  idxA)  = Sigma3_tilde(idxA,  idxA)  + M33;

    % Exact Hermitian sensing coefficient.
    Sigma3 = (2 * Pre.L / Pre.noise_s) * (Sigma3_tilde + Sigma3_tilde') / 2;
end


function out=Shift_parameter_appromate(C2)
D=diag(C2);
R = sum(abs(C2), 2) - abs(D);
out = max(D + R);
end


function Y = expjAngle(X)
% Return exp(1j * angle(X)) only at nonzero entries, zero elsewhere
PhaseX = angle(X);
PhaseX(X == 0) = 0; % optional, not strictly necessary
Y = exp(1j * PhaseX) .* (X ~= 0);
end


function FIM=calculateFIM(Pre, D)

%CALCULATEFIM Compute the real-valued FIM using precomputed matrices
%
% Inputs:
%   Pre : struct containing all precomputed parameters
%       Required fields:
%           Pre.L
%           Pre.noise_s
%           Pre.tilde_A
%           Pre.tilde_A_o
%           Pre.tilde_A_th
%           Pre.tilde_A_ph
%           Pre.tilde_B_oo
%           Pre.tilde_B_oth
%           Pre.tilde_B_oph
%           Pre.tilde_B_tho
%           Pre.tilde_B_thth
%           Pre.tilde_B_thph
%           Pre.tilde_B_pho
%           Pre.tilde_B_phth
%           Pre.tilde_B_phph
%
%   D   : reduced beamforming matrix, with W = U*D
%
% Output:
%   FIM : 4M x 4M real-valued Fisher information matrix

% Precompute projected beamformers
Wo  = Pre.tilde_A_o  * D;   % M x Ns
Wth = Pre.tilde_A_th * D;   % M x Ns
Wph = Pre.tilde_A_ph * D;   % M x Ns
Wa  = Pre.tilde_A    * D;   % M x Nsin

% Reusable correlation terms:
% (A_x * D) * (A_y * D)' = A_x * (D*D') * A_y'
Goo  = Wo  * Wo';    % M x M
Goth = Wo  * Wth';   % M x M
Gtho = Wth * Wo';    % M x M
Goph = Wo  * Wph';   % M x M
Gpho = Wph * Wo';    % M x M
Gthth = Wth * Wth';  % M x M
Gphph = Wph * Wph';  % M x M
Gath = Wa  * Wth';   % M x M
Gao  = Wa  * Wo';    % M x M
Gaph = Wa  * Wph';   % M x M
Gaa  = Wa  * Wa';    % M x M

Gphth = Wph * Wth';

% F11
F11 = (Goo).'   .* Pre.tilde_B_thth + ...
    (Goth).' .* Pre.tilde_B_oth  + ...
    (Gtho).' .* Pre.tilde_B_tho +...
    (Gthth).' .* Pre.tilde_B_oo;


F12 = (Goo).'   .* Pre.tilde_B_thph + ...
    (Goth).'  .* Pre.tilde_B_oph  + ...
    (Gpho).'  .* Pre.tilde_B_tho  + ...
    (Gphth).' .* Pre.tilde_B_oo;

% F13
F13 = (Gao).'   .* Pre.tilde_B_tho + ...
    (Gath).'  .* Pre.tilde_B_oo;


% F22
F22 = (Goo).'   .* Pre.tilde_B_phph + ...
    (Goph).' .* Pre.tilde_B_oph  + ...
    (Gpho).' .* Pre.tilde_B_pho +...
    (Gphph).' .* Pre.tilde_B_oo;
% F23
F23 = (Gao).'   .* Pre.tilde_B_pho + ...
    (Gaph).'  .* Pre.tilde_B_oo;

% F33
F33 = (Gaa).'   .* Pre.tilde_B_oo;


M = size(Pre.tilde_B_oo, 1);
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

% Final scaling
FIM = 2 * Pre.L / Pre.noise_s * FIM;

end


function [A,dAtheta,dAphi]=construct_steer_matrix_and_derivative_steer_matrix(theta,phi,Mx,My)
% input:
% theta - Elevation angle (in radians)
% phi   - Azimuth angle (in radians)
% Mx    - Number of elements along x-axis
% My    - Number of elements along y-axis

% Generate indices for x-axis and y-axis
ix = (0:Mx-1)'; % Column vector for x-axis indices
iy = (0:My-1)'; % Column vector for y-axis indices

% Number of steering vectors
M=length(theta);
% Preallocate outputs
A=zeros(Mx*My,M);
dAtheta=zeros(Mx*My,M);
dAphi=zeros(Mx*My,M);
% Compute steering vectors along x and y axes
for m=1:M
    % Steering vectors for x and y dimensions
    ax =1/sqrt(Mx)* exp(1j * pi * ix * sin(theta(m)) * sin(phi(m)));
    ay =1/sqrt(My)* exp(1j * pi * iy * cos(phi(m)));
    % Derivatives of ax and ay
    daxtheta=1j*pi*ix*cos(theta(m))*sin(phi(m)).*ax;

    daxphi=1j*pi*ix*sin(theta(m))*cos(phi(m)).*ax;
    dayphi=-1j*pi*iy*sin(phi(m)).*ay;
    % Compute the full UPA steering vector using Kronecker product
    A(:,m) = kron(ay, ax); % Kronecker product combines ax and ay
    dAtheta(:,m)=kron(ay,daxtheta);
    dAphi(:,m)=kron(ay,daxphi)+kron(dayphi,ax);
end

end


function [obj, Wa, Wd, iter_used] = matrix_approximation_groupconnected(W, Wa, Wd, Z)
%MATRIX_APPROXIMATION_GROUPCONNECTED Mask-aware AO for GC-HBF.
%
% Approximately solves
%   min_{Wa,Wd} ||W-Wa*Wd||_F^2
% subject to
%   Wa = Z .* exp(1j*Psi).
%
% The analog MM update is valid for an arbitrary fixed binary connectivity
% mask. Unlike the SC special case, Wa'*Wa is generally not diagonal for
% G < N_RF, so the digital step uses the exact least-squares solution.

    count_max  = 200;
    inner_iter = 20;
    tol_outer  = 1e-5;
    tol_inner  = 1e-4;

    obj = zeros(count_max, 1);
    Wa = Z .* exp(1j * angle(Wa));

    for count = 1:count_max

        % ----- Exact digital LS update for the current GC analog matrix -----
        Wd = digital_update_groupconnected(Wa, W);

        % ----- Mask-aware analog MM update -----
        Rwd = Wd * Wd';
        Rwd = (Rwd + Rwd') / 2;
        mu1 = max(real(eig(Rwd)));

        for iter = 1:inner_iter
            Wa_old = Wa;

            D = W * Wd' + Wa * (mu1 * eye(size(Rwd)) - Rwd);
            Wa = Z .* exp(1j * angle(D));

            rel_inner = norm(Wa - Wa_old, 'fro') / ...
                        max(norm(Wa_old, 'fro'), 1e-12);
            if rel_inner < tol_inner
                break;
            end
        end

        E = W - Wa * Wd;
        obj(count) = norm(E, 'fro')^2;

        if count > 1
            rel_outer = abs(obj(count) - obj(count-1)) / ...
                        max(obj(count-1), 1e-12);
            if rel_outer < tol_outer
                obj = obj(1:count);
                iter_used = count;
                return;
            end
        end
    end

    iter_used = count_max;
end


function Wd = digital_update_groupconnected(Wa, Wtarget)
%DIGITAL_UPDATE_GROUPCONNECTED Exact digital least-squares update.
%
% Solves
%   min_Wd ||Wtarget-Wa*Wd||_F^2.
%
% For GC/FC Wa'*Wa is not generally diagonal, so use MATLAB's stable
% least-squares backslash rather than the SC-only diagonal formula.

    Wd = Wa \ Wtarget;
end


function Z = build_groupconnected_mask(Nt, N_RF, G)
%BUILD_GROUPCONNECTED_MASK Contiguous balanced GC connectivity mask.
%
% The antennas and RF chains are partitioned into G groups. Every antenna
% in a group is connected to every RF chain in the same group, and there
% are no cross-group connections.
%
% Special cases:
%   G = 1      -> fully connected
%   G = N_RF   -> sub-connected, when Nt is divisible by N_RF

    if G < 1 || G > min(Nt, N_RF) || round(G) ~= G
        error('G must be an integer satisfying 1 <= G <= min(Nt,N_RF).');
    end

    if mod(Nt, G) ~= 0 || mod(N_RF, G) ~= 0
        error('For this balanced GC construction, G must divide Nt and N_RF.');
    end

    Nant_g = Nt / G;
    Nrf_g  = N_RF / G;

    Z = zeros(Nt, N_RF);

    for g = 1:G
        ant_idx = (g-1)*Nant_g + (1:Nant_g);
        rf_idx  = (g-1)*Nrf_g  + (1:Nrf_g);
        Z(ant_idx, rf_idx) = 1;
    end
end


function y = db2power(x)
%DB2POWER Convert dB to linear power ratio without a toolbox.
    y = 10.^(x / 10);
end


function y = square_abs(x)
%SQUARE_ABS Element-wise squared magnitude.
    y = abs(x).^2;
end
