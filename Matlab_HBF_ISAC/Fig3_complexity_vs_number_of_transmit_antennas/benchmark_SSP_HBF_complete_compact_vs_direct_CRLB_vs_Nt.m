clc
clear
close all

%% ================================================================
% COMPLETE SSP-HBF COMPLEXITY BENCHMARK VERSUS Nt
%
% Proposed compact implementation versus previous direct-matrix
% implementation.
%
% IMPORTANT:
% The comparison includes the WHOLE objective/gradient pipeline:
%
% Proposed compact implementation:
%   1) compact FIM/CRLB evaluation without Rx = W*W'
%   2) small Sigma3 sensing coefficient
%   3) compact gradient N*(Omega1 + Omega2*N'*W)
%   4) reduced-dimensional dominant shift
%   5) fixed-cost GC-HBF decomposition
%   6) compact candidate FIM/CRLB + SR evaluation
%
% Direct implementation:
%   1) explicitly construct Rx = W*W' (Nt x Nt)
%   2) original full-matrix FIM/CRLB evaluation
%   3) explicitly construct full sensing Q (Nt x Nt)
%   4) explicitly construct C2 (Nt x Nt)
%   5) dominant eigenvalue of full C2
%   6) fixed-cost GC-HBF decomposition (IDENTICAL to compact method)
%   7) original direct candidate FIM/CRLB + SR evaluation
%
% Thus the only difference is the algebraic implementation of the
% ISAC objective/gradient/shift. The HBF projection is held identical.
%
% Timing is SERIAL deliberately. PARFOR/worker contention should not be
% used for a runtime-scaling figure.
% ================================================================

%% ================================================================
% Common sensing realizations
% ================================================================
load fixed_dataset_K8_Nt64_M3.mat

%% ================================================================
% Same weight as the convergence experiment
% ================================================================
rho_tradeoff = 1e-3;

delta_c = rho_tradeoff/(1+rho_tradeoff);
delta_s = 1/(1+rho_tradeoff);
lambda  = delta_s/delta_c;                 % = 1000

fprintf('\nFixed tradeoff weight:\n');
fprintf('rho = %.3e\n',rho_tradeoff);
fprintf('delta_c = %.12e\n',delta_c);
fprintf('delta_s = %.12e\n',delta_s);
fprintf('lambda = %.1f\n\n',lambda);

%% ================================================================
% Fixed system dimensions
% ================================================================
M = 3;
K = 8;
N_RF = K;

% IMPORTANT: keep receive array fixed while varying Nt.
Nrh = 10;
Nrv = 8;
Nr = Nrh*Nrv;                              % 80

Pt      = db2power(20);
noise_c = db2power(0);
noise_s = db2power(0);
L = 64;

%% ================================================================
% Transmit-array sweep
%
% 1024 is intentionally the maximum. Larger arrays are unnecessary for
% this practical scaling experiment.
% ================================================================
Nt_list  = [32, 64, 128, 256, 512, 1024];
Nth_list = [ 8,  8,  16,  16,  32,   32];
Ntv_list = [ 4,  8,   8,  16,  16,   32];

if any(Nth_list.*Ntv_list ~= Nt_list)
    error('Nth_list .* Ntv_list must equal Nt_list.');
end

%% ================================================================
% Representative GC architecture
% ================================================================
G_test = 4;

if mod(N_RF,G_test) ~= 0 || any(mod(Nt_list,G_test) ~= 0)
    error('G_test must divide N_RF and all Nt values.');
end

%% ================================================================
% Deterministic timing parameters
% ================================================================
I_timing = 10;

% Measure several identical calls inside ONE tic/toc block, then divide.
% This avoids timer overhead contaminating the very fast compact method.
repeat_fim  = 50;
repeat_core = 10;
repeat_full = 3;

% Fixed-cost hybrid decomposition.
% These are held exactly equal for compact and direct implementations.
num_AO_outer = 5;
num_AO_inner = 5;

mu_floor  = 1e-8;
mu_margin = 1e-8;

% Direct full-matrix EIGS settings.
% NOTE: no opts.issym field -- that option caused the previous warning
% when an explicit matrix was supplied to eigs.
eigs_tol   = 1e-10;
eigs_maxit = 1000;

%% ================================================================
% Reproducible nested communication channels
% ================================================================
if size(alpha_test,1) < I_timing || ...
        size(theta_test,1) < I_timing || ...
        size(phi_test,1) < I_timing
    error('The dataset must contain at least %d realizations.',I_timing);
end

rng(20260910,'twister');

Nt_max = max(Nt_list);

H_master = 1/sqrt(2) * ...
    (randn(I_timing,Nt_max,K) + ...
     1j*randn(I_timing,Nt_max,K));

%% ================================================================
% Storage
% ================================================================
Nsize = length(Nt_list);

T_fim_compact_all = nan(I_timing,Nsize);
T_fim_direct_all  = nan(I_timing,Nsize);

T_core_compact_all = nan(I_timing,Nsize);
T_core_direct_all  = nan(I_timing,Nsize);

T_full_compact_all = nan(I_timing,Nsize);
T_full_direct_all  = nan(I_timing,Nsize);

RelFIMErr_all    = nan(I_timing,Nsize);
RelTargetErr_all = nan(I_timing,Nsize);
RelMuErr_all     = nan(I_timing,Nsize);

% Complex-double memory of ONE explicitly formed Nt x Nt matrix.
DirectMatrixMemory_MB = 16*Nt_list.^2/1024^2;

save_dir = 'complexity_results_SSP_HBF_complete';
fig_dir  = 'complexity_figures_SSP_HBF_complete';

if ~exist(save_dir,'dir')
    mkdir(save_dir);
end

if ~exist(fig_dir,'dir')
    mkdir(fig_dir);
end

save_file = fullfile(save_dir,sprintf( ...
    'data_complexity_complete_SSP_HBF_G%d_rho1e_3.mat',G_test));

%% ================================================================
% Main Nt loop
% ================================================================
for iN = 1:Nsize

    Nt  = Nt_list(iN);
    Nth = Nth_list(iN);
    Ntv = Ntv_list(iN);

    Z = build_groupconnected_mask(Nt,N_RF,G_test);

    fprintf('\n============================================================\n');
    fprintf('Nt = %d (%d x %d), Nr = %d fixed, G = %d\n', ...
        Nt,Nth,Ntv,Nr,G_test);
    fprintf('One dense complex Nt x Nt matrix = %.2f MB\n', ...
        DirectMatrixMemory_MB(iN));
    fprintf('============================================================\n');

    for channel = 1:I_timing

        fprintf('Nt=%4d | realization %2d/%2d\n', ...
            Nt,channel,I_timing);

        %% --------------------------------------------------------
        % Paired channel realization
        % ---------------------------------------------------------
        H = squeeze(H_master(channel,1:Nt,:));

        alpha = double(alpha_test(channel,1:M).');
        theta = double(theta_test(channel,1:M).');
        phi   = double(phi_test(channel,1:M).');

        [A,dAtheta,dAphi] = ...
            construct_steer_matrix_and_derivative_steer_matrix( ...
            theta,phi,Nth,Ntv);

        [B,dBtheta,dBphi] = ...
            construct_steer_matrix_and_derivative_steer_matrix( ...
            theta,phi,Nrh,Nrv);

        Gamma = diag(alpha);

        %% --------------------------------------------------------
        % Compact precomputed sensing quantities
        % ---------------------------------------------------------
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

        %% --------------------------------------------------------
        % Compact range basis
        %
        % Nbasis and its QR depend on the channel/angles, NOT on the SSP
        % iteration. Therefore they are precomputed and excluded from
        % per-iteration timing in both methods.
        % ---------------------------------------------------------
        Nbasis = [H,A,dAtheta,dAphi];
        [~,R_N,~] = qr_rank_basis(Nbasis);

        %% --------------------------------------------------------
        % Same GC-feasible starting beamformer for both methods
        % ---------------------------------------------------------
        Wa = Z.*exp(1j*angle(H(:,1:N_RF)));

        Wd = digital_update_groupconnected(Wa,H);
        Wd = sqrt(Pt)/max(norm(Wa*Wd,'fro'),1e-12)*Wd;

        W = Wa*Wd;

        %% ========================================================
        % EQUIVALENCE CHECK BEFORE TIMING
        % =========================================================
        FIM_c = calculateFIM(Pre,W);

        FIM_d = calculateFIM_direct( ...
            L,noise_s,W,A,dAtheta,dAphi, ...
            B,dBtheta,dBphi,Gamma);

        RelFIMErr_all(channel,iN) = ...
            norm(FIM_c-FIM_d,'fro') / ...
            max(norm(FIM_d,'fro'),1e-12);

        [Wtar_c,mu_c] = compact_core( ...
            W,H,Nbasis,R_N,Pre,noise_c, ...
            delta_c,delta_s,mu_floor,mu_margin);

        [Wtar_d,mu_d] = direct_core( ...
            W,H,A,dAtheta,dAphi, ...
            B,dBtheta,dBphi,Gamma, ...
            L,noise_s,noise_c, ...
            delta_c,delta_s, ...
            mu_floor,mu_margin,eigs_tol,eigs_maxit);

        RelTargetErr_all(channel,iN) = ...
            norm(Wtar_c-Wtar_d,'fro') / ...
            max(norm(Wtar_d,'fro'),1e-12);

        RelMuErr_all(channel,iN) = ...
            abs(mu_c-mu_d) / max(abs(mu_d),1e-12);

        if RelFIMErr_all(channel,iN) > 1e-6 || ...
                RelTargetErr_all(channel,iN) > 1e-6

            warning(['Compact/direct mismatch at Nt=%d, channel=%d: ' ...
                'FIM %.3e, target %.3e'], ...
                Nt,channel, ...
                RelFIMErr_all(channel,iN), ...
                RelTargetErr_all(channel,iN));
        end

        %% ========================================================
        % JIT / BLAS WARM-UP
        % =========================================================
        compact_core( ...
            W,H,Nbasis,R_N,Pre,noise_c, ...
            delta_c,delta_s,mu_floor,mu_margin);

        direct_core( ...
            W,H,A,dAtheta,dAphi, ...
            B,dBtheta,dBphi,Gamma, ...
            L,noise_s,noise_c, ...
            delta_c,delta_s, ...
            mu_floor,mu_margin,eigs_tol,eigs_maxit);

        full_iteration_compact( ...
            W,Wa,Wd,Z,H,Nbasis,R_N,Pre,noise_c,Pt, ...
            delta_c,delta_s,mu_floor,mu_margin, ...
            num_AO_outer,num_AO_inner);

        full_iteration_direct( ...
            W,Wa,Wd,Z,H,A,dAtheta,dAphi, ...
            B,dBtheta,dBphi,Gamma, ...
            L,noise_s,noise_c,Pt, ...
            delta_c,delta_s,mu_floor,mu_margin, ...
            num_AO_outer,num_AO_inner,eigs_tol,eigs_maxit);

        %% ========================================================
        % TEST A: FIM + CRLB ONLY
        %
        % This explicitly verifies the linearized CRLB/FIM evaluation.
        % =========================================================
        t0 = tic;

        for rep = 1:repeat_fim
            Ftmp = calculateFIM(Pre,W);
            CRtmp = trace(inv(Ftmp)); %#ok<NASGU>
        end

        T_fim_compact_all(channel,iN) = ...
            toc(t0)/repeat_fim;

        t0 = tic;

        for rep = 1:repeat_fim
            Ftmp = calculateFIM_direct( ...
                L,noise_s,W,A,dAtheta,dAphi, ...
                B,dBtheta,dBphi,Gamma);
            CRtmp = trace(inv(Ftmp)); %#ok<NASGU>
        end

        T_fim_direct_all(channel,iN) = ...
            toc(t0)/repeat_fim;

        %% ========================================================
        % TEST B: COMPLETE ISAC CORE
        %
        % Includes FIM/CRLB + sensing coefficient + communication
        % auxiliaries + gradient + shift + target construction.
        % =========================================================
        t0 = tic;

        for rep = 1:repeat_core
            compact_core( ...
                W,H,Nbasis,R_N,Pre,noise_c, ...
                delta_c,delta_s,mu_floor,mu_margin);
        end

        T_core_compact_all(channel,iN) = ...
            toc(t0)/repeat_core;

        t0 = tic;

        for rep = 1:repeat_core
            direct_core( ...
                W,H,A,dAtheta,dAphi, ...
                B,dBtheta,dBphi,Gamma, ...
                L,noise_s,noise_c, ...
                delta_c,delta_s, ...
                mu_floor,mu_margin,eigs_tol,eigs_maxit);
        end

        T_core_direct_all(channel,iN) = ...
            toc(t0)/repeat_core;

        %% ========================================================
        % TEST C: COMPLETE NOMINAL SSP-HBF OUTER ITERATION
        %
        % Core + identical HBF decomposition + candidate objective.
        % =========================================================
        t0 = tic;

        for rep = 1:repeat_full
            full_iteration_compact( ...
                W,Wa,Wd,Z,H,Nbasis,R_N,Pre,noise_c,Pt, ...
                delta_c,delta_s,mu_floor,mu_margin, ...
                num_AO_outer,num_AO_inner);
        end

        T_full_compact_all(channel,iN) = ...
            toc(t0)/repeat_full;

        t0 = tic;

        for rep = 1:repeat_full
            full_iteration_direct( ...
                W,Wa,Wd,Z,H,A,dAtheta,dAphi, ...
                B,dBtheta,dBphi,Gamma, ...
                L,noise_s,noise_c,Pt, ...
                delta_c,delta_s,mu_floor,mu_margin, ...
                num_AO_outer,num_AO_inner,eigs_tol,eigs_maxit);
        end

        T_full_direct_all(channel,iN) = ...
            toc(t0)/repeat_full;
    end

    %% ------------------------------------------------------------
    % Checkpoint after every Nt
    % -------------------------------------------------------------
    save(save_file, ...
        'rho_tradeoff','delta_c','delta_s','lambda', ...
        'M','K','N_RF','Nr','Nrh','Nrv','G_test', ...
        'Pt','noise_c','noise_s','L', ...
        'Nt_list','Nth_list','Ntv_list', ...
        'I_timing','repeat_fim','repeat_core','repeat_full', ...
        'num_AO_outer','num_AO_inner', ...
        'T_fim_compact_all','T_fim_direct_all', ...
        'T_core_compact_all','T_core_direct_all', ...
        'T_full_compact_all','T_full_direct_all', ...
        'RelFIMErr_all','RelTargetErr_all','RelMuErr_all', ...
        'DirectMatrixMemory_MB','-v7.3');
end

%% ================================================================
% Aggregate using median across realizations
% ================================================================
T_fim_compact  = median(T_fim_compact_all,1,'omitnan');
T_fim_direct   = median(T_fim_direct_all,1,'omitnan');

T_core_compact = median(T_core_compact_all,1,'omitnan');
T_core_direct  = median(T_core_direct_all,1,'omitnan');

T_full_compact = median(T_full_compact_all,1,'omitnan');
T_full_direct  = median(T_full_direct_all,1,'omitnan');

Speedup_fim  = T_fim_direct./T_fim_compact;
Speedup_core = T_core_direct./T_core_compact;
Speedup_full = T_full_direct./T_full_compact;

%% ================================================================
% Linear fits T = a*Nt+b for proposed implementation
% ================================================================
[coef_fim_compact,R2_fim_compact] = ...
    fit_linear(Nt_list,T_fim_compact);

[coef_core_compact,R2_core_compact] = ...
    fit_linear(Nt_list,T_core_compact);

[coef_full_compact,R2_full_compact] = ...
    fit_linear(Nt_list,T_full_compact);

%% ================================================================
% Diagnostics
% ================================================================
fprintf('\n============================================================\n');
fprintf('COMPLETE COMPLEXITY BENCHMARK RESULTS\n');
fprintf('============================================================\n');

fprintf('\nCompact linear fits T = a Nt + b:\n');
fprintf('FIM+CRLB : R^2 = %.4f\n',R2_fim_compact);
fprintf('ISAC core: R^2 = %.4f\n',R2_core_compact);
fprintf('Full SSP : R^2 = %.4f\n',R2_full_compact);

fprintf('\nMaximum algebraic mismatch:\n');
fprintf('FIM relative error    = %.3e\n', ...
    max(RelFIMErr_all(:),[],'omitnan'));
fprintf('Target relative error = %.3e\n', ...
    max(RelTargetErr_all(:),[],'omitnan'));
fprintf('Shift relative error  = %.3e\n', ...
    max(RelMuErr_all(:),[],'omitnan'));

fprintf('\n Nt   FIM comp  FIM direct   core comp core direct  full comp full direct\n');

for iN = 1:Nsize
    fprintf('%4d  %8.3f  %10.3f   %8.3f %10.3f   %8.3f %10.3f  ms\n', ...
        Nt_list(iN), ...
        1e3*T_fim_compact(iN), ...
        1e3*T_fim_direct(iN), ...
        1e3*T_core_compact(iN), ...
        1e3*T_core_direct(iN), ...
        1e3*T_full_compact(iN), ...
        1e3*T_full_direct(iN));
end

fprintf('\nSpeedup at Nt=%d:\n',Nt_list(end));
fprintf('FIM+CRLB : %.2fx\n',Speedup_fim(end));
fprintf('ISAC core: %.2fx\n',Speedup_core(end));
fprintf('Full SSP : %.2fx\n',Speedup_full(end));

%% ================================================================
% Final save
% ================================================================
save(save_file, ...
    'rho_tradeoff','delta_c','delta_s','lambda', ...
    'M','K','N_RF','Nr','Nrh','Nrv','G_test', ...
    'Pt','noise_c','noise_s','L', ...
    'Nt_list','Nth_list','Ntv_list', ...
    'I_timing','repeat_fim','repeat_core','repeat_full', ...
    'num_AO_outer','num_AO_inner', ...
    'T_fim_compact_all','T_fim_direct_all', ...
    'T_core_compact_all','T_core_direct_all', ...
    'T_full_compact_all','T_full_direct_all', ...
    'T_fim_compact','T_fim_direct', ...
    'T_core_compact','T_core_direct', ...
    'T_full_compact','T_full_direct', ...
    'Speedup_fim','Speedup_core','Speedup_full', ...
    'coef_fim_compact','R2_fim_compact', ...
    'coef_core_compact','R2_core_compact', ...
    'coef_full_compact','R2_full_compact', ...
    'RelFIMErr_all','RelTargetErr_all','RelMuErr_all', ...
    'DirectMatrixMemory_MB','-v7.3');

%% ================================================================
% FIGURE 1: FIM + CRLB evaluation
% ================================================================
fig1 = figure('Color','w');
hold on;

plot(Nt_list,1e3*T_fim_compact, ...
    '-o','LineWidth',1.6,'MarkerSize',5);

plot(Nt_list,1e3*T_fim_direct, ...
    '--s','LineWidth',1.6,'MarkerSize',5);

grid on;
box on;
xlabel('Number of transmit antennas, N_t');
ylabel('FIM/CRLB evaluation time (ms)');
legend({'Compact','Direct matrix'},'Location','northwest');

set_ieee_figure_style(fig1);
savefig(fig1,fullfile(fig_dir,'runtime_FIM_CRLB_vs_Nt.fig'));

%% ================================================================
% FIGURE 2: Complete ISAC objective/gradient/shift core
% ================================================================
fig2 = figure('Color','w');
hold on;

plot(Nt_list,1e3*T_core_compact, ...
    '-o','LineWidth',1.6,'MarkerSize',5);

plot(Nt_list,1e3*T_core_direct, ...
    '--s','LineWidth',1.6,'MarkerSize',5);

grid on;
box on;
xlabel('Number of transmit antennas, N_t');
ylabel('ISAC core update time (ms)');
legend({'Compact','Direct matrix'},'Location','northwest');

set_ieee_figure_style(fig2);
savefig(fig2,fullfile(fig_dir,'runtime_ISAC_core_vs_Nt.fig'));

%% ================================================================
% FIGURE 3: Complete SSP-HBF outer iteration
% ================================================================
fig3 = figure('Color','w');
hold on;

plot(Nt_list,1e3*T_full_compact, ...
    '-o','LineWidth',1.6,'MarkerSize',5);

plot(Nt_list,1e3*T_full_direct, ...
    '--s','LineWidth',1.6,'MarkerSize',5);

grid on;
box on;
xlabel('Number of transmit antennas, N_t');
ylabel('Runtime per SSP-HBF iteration (ms)');
legend({'Compact SSP-HBF','Direct-matrix SSP-HBF'}, ...
    'Location','northwest');

set_ieee_figure_style(fig3);
savefig(fig3,fullfile(fig_dir,'runtime_complete_SSP_iteration_vs_Nt.fig'));

%% ================================================================
% FIGURE 4: Speedup
% ================================================================
fig4 = figure('Color','w');
hold on;

plot(Nt_list,Speedup_fim, ...
    '-o','LineWidth',1.5,'MarkerSize',5);

plot(Nt_list,Speedup_core, ...
    '-s','LineWidth',1.5,'MarkerSize',5);

plot(Nt_list,Speedup_full, ...
    '-d','LineWidth',1.5,'MarkerSize',5);

grid on;
box on;
xlabel('Number of transmit antennas, N_t');
ylabel('Runtime speedup');
legend({'FIM/CRLB','ISAC core','Full SSP-HBF'}, ...
    'Location','northwest');

set_ieee_figure_style(fig4);
savefig(fig4,fullfile(fig_dir,'runtime_speedup_vs_Nt.fig'));

fprintf('\nSaved data to:\n%s\n',save_file);
fprintf('Figures saved in:\n%s\n',fig_dir);

%% ================================================================
% COMPLETE COMPACT/DIRECT COMPUTATIONAL CORES
% ================================================================

function [W_target,mu,FIM,CRLB,SR] = compact_core( ...
    W,H,Nbasis,R_N,Pre,noise_c, ...
    delta_c,delta_s,mu_floor,mu_margin)
%COMPACT_CORE Entire proposed compact objective/gradient/shift pipeline.

    K = size(H,2);
    M = size(Pre.tilde_B_oo,1);

    %% compact FIM / CRLB
    FIM = calculateFIM(Pre,W);
    CRBM = inv(FIM);
    CRLB = trace(CRBM);

    %% communication auxiliary variables
    T_k = sum(square_abs(H'*W),2) + noise_c*ones(K,1);

    signal_power = square_abs(diag(H'*W));
    rate_p = log(T_k./(T_k-signal_power));
    SR = sum(rate_p);

    alpha_k = T_k./(T_k-signal_power)-1;
    beta_k = sqrt(1+alpha_k).*diag(H'*W)./T_k;

    Sigma1 = diag(sqrt(1+alpha_k).*beta_k);
    Sigma2 = diag(square_abs(beta_k));

    %% compact sensing coefficient
    Phi = CRBM*CRBM;
    Sigma3 = construct_Sigma3(Pre,Phi);

    Omega1 = [delta_c*Sigma1; zeros(3*M,K)];
    Omega2 = blkdiag(-delta_c*Sigma2,delta_s*Sigma3);

    %% reduced-dimensional shift
    [mu_exact,~] = ...
        compute_lowdim_exact_shift_from_Omega2(R_N,Omega2);

    mu = max(mu_exact+mu_margin,mu_floor);

    %% compact gradient and target
    grad_total = Nbasis * ...
        (Omega1 + Omega2*(Nbasis'*W));

    W_target = grad_total + mu*W;
end


function [W_target,mu,FIM,CRLB,SR] = direct_core( ...
    W,H,A,dAtheta,dAphi,B,dBtheta,dBphi,Gamma, ...
    L,noise_s,noise_c,delta_c,delta_s, ...
    mu_floor,mu_margin,eigs_tol,eigs_maxit)
%DIRECT_CORE Previous direct Nt-by-Nt implementation.

    K = size(H,2);
    Nt = size(W,1);

    %% Original/direct FIM explicitly uses Rx = W*W'
    FIM = calculateFIM_direct( ...
        L,noise_s,W,A,dAtheta,dAphi, ...
        B,dBtheta,dBphi,Gamma);

    CRBM = inv(FIM);
    CRLB = trace(CRBM);

    %% communication auxiliary variables
    T_k = sum(square_abs(H'*W),2) + noise_c*ones(K,1);

    signal_power = square_abs(diag(H'*W));
    rate_p = log(T_k./(T_k-signal_power));
    SR = sum(rate_p);

    alpha_k = T_k./(T_k-signal_power)-1;
    beta_k = sqrt(1+alpha_k).*diag(H'*W)./T_k;

    Sigma1 = diag(sqrt(1+alpha_k).*beta_k);
    Sigma2 = diag(square_abs(beta_k));

    %% Original/direct sensing gradient matrix Q: Nt x Nt
    Phi = CRBM*CRBM;

    Q = construct_matrixQ_direct( ...
        L,noise_s,Phi, ...
        A,dAtheta,dAphi, ...
        B,dBtheta,dBphi,Gamma);

    %% Explicit full quadratic coefficient C2: Nt x Nt
    C1 = delta_c*H*Sigma1;

    C2 = 0.5*delta_s*(Q+Q') ...
        - delta_c*H*Sigma2*H';

    C2 = (C2+C2')/2;

    %% Full-dimensional dominant eigenvalue
    opts = struct();
    opts.tol = eigs_tol;
    opts.maxit = eigs_maxit;
    opts.v0 = ones(Nt,1)/sqrt(Nt);

    try
        mu_exact = abs(eigs(C2,1,'LM',opts));
    catch
        % Only a numerical fallback. It still uses the full Nt x Nt C2.
        ev = real(eig(C2));
        mu_exact = max(abs(ev));
    end

    mu = max(mu_exact+mu_margin,mu_floor);

    %% Direct target
    W_target = C1 + C2*W + mu*W;
end


%% ================================================================
% COMPLETE NOMINAL SSP-HBF ITERATIONS
% ================================================================

function [W_new,Wa_new,Wd_new,obj_new] = full_iteration_compact( ...
    W,Wa,Wd,Z,H,Nbasis,R_N,Pre,noise_c,Pt, ...
    delta_c,delta_s,mu_floor,mu_margin, ...
    num_AO_outer,num_AO_inner)

    [W_target,~,~,~,~] = compact_core( ...
        W,H,Nbasis,R_N,Pre,noise_c, ...
        delta_c,delta_s,mu_floor,mu_margin);

    [Wa_new,Wd_new] = matrix_approximation_fixed( ...
        W_target,Wa,Wd,Z,num_AO_outer,num_AO_inner);

    Wd_new = sqrt(Pt)/ ...
        max(norm(Wa_new*Wd_new,'fro'),1e-12)*Wd_new;

    W_new = Wa_new*Wd_new;

    % Candidate objective evaluation ALSO uses compact FIM.
    FIM_new = calculateFIM(Pre,W_new);
    CRLB_new = trace(inv(FIM_new));

    K = size(H,2);
    T_k = sum(square_abs(H'*W_new),2) + noise_c*ones(K,1);
    SR_new = sum(log(T_k./ ...
        (T_k-square_abs(diag(H'*W_new)))));

    obj_new = delta_c*SR_new-delta_s*CRLB_new;
end


function [W_new,Wa_new,Wd_new,obj_new] = full_iteration_direct( ...
    W,Wa,Wd,Z,H,A,dAtheta,dAphi,B,dBtheta,dBphi,Gamma, ...
    L,noise_s,noise_c,Pt,delta_c,delta_s, ...
    mu_floor,mu_margin,num_AO_outer,num_AO_inner, ...
    eigs_tol,eigs_maxit)

    [W_target,~,~,~,~] = direct_core( ...
        W,H,A,dAtheta,dAphi,B,dBtheta,dBphi,Gamma, ...
        L,noise_s,noise_c,delta_c,delta_s, ...
        mu_floor,mu_margin,eigs_tol,eigs_maxit);

    [Wa_new,Wd_new] = matrix_approximation_fixed( ...
        W_target,Wa,Wd,Z,num_AO_outer,num_AO_inner);

    Wd_new = sqrt(Pt)/ ...
        max(norm(Wa_new*Wd_new,'fro'),1e-12)*Wd_new;

    W_new = Wa_new*Wd_new;

    % Candidate objective evaluation uses the ORIGINAL direct FIM.
    FIM_new = calculateFIM_direct( ...
        L,noise_s,W_new,A,dAtheta,dAphi, ...
        B,dBtheta,dBphi,Gamma);

    CRLB_new = trace(inv(FIM_new));

    K = size(H,2);
    T_k = sum(square_abs(H'*W_new),2) + noise_c*ones(K,1);
    SR_new = sum(log(T_k./ ...
        (T_k-square_abs(diag(H'*W_new)))));

    obj_new = delta_c*SR_new-delta_s*CRLB_new;
end


%% ================================================================
% IDENTICAL FIXED-COST HBF DECOMPOSITION FOR BOTH METHODS
% ================================================================

function [Wa,Wd] = matrix_approximation_fixed( ...
    Wtarget,Wa,Wd,Z,num_outer,num_inner)

    Wa = Z.*exp(1j*angle(Wa));

    for io = 1:num_outer

        Wd = digital_update_groupconnected(Wa,Wtarget);

        Rwd = Wd*Wd';
        Rwd = (Rwd+Rwd')/2;

        mu1 = max(real(eig(Rwd)));

        for ii = 1:num_inner

            D = Wtarget*Wd' + ...
                Wa*(mu1*eye(size(Rwd))-Rwd);

            Wa = Z.*exp(1j*angle(D));
        end
    end

    Wd = digital_update_groupconnected(Wa,Wtarget);
end


%% ================================================================
% Small utilities
% ================================================================

function [coef,R2] = fit_linear(x,y)

    valid = isfinite(x) & isfinite(y);

    x = double(x(valid));
    y = double(y(valid));

    coef = polyfit(x,y,1);
    yhat = polyval(coef,x);

    SSres = sum((y-yhat).^2);
    SStot = sum((y-mean(y)).^2);

    R2 = 1-SSres/max(SStot,eps);
end


function set_ieee_figure_style(fig_handle)

    fig_width = 3.5;
    fig_height = 2.65;

    set(fig_handle, ...
        'Units','inches', ...
        'Position',[1 1 fig_width fig_height], ...
        'Color','w');

    set(fig_handle, ...
        'PaperUnits','inches', ...
        'PaperPosition',[0 0 fig_width fig_height], ...
        'PaperSize',[fig_width fig_height]);

    ax = findall(fig_handle,'Type','axes');

    for k = 1:length(ax)
        set(ax(k), ...
            'FontName','Times New Roman', ...
            'FontSize',9, ...
            'LineWidth',0.8, ...
            'Box','on');
    end

    lgd = findall(fig_handle,'Type','legend');

    for k = 1:length(lgd)
        set(lgd(k), ...
            'FontName','Times New Roman', ...
            'FontSize',8, ...
            'Box','off');
    end
end


function y = db2power(x)
    y = 10.^(x/10);
end


function y = square_abs(x)
    y = abs(x).^2;
end




%% ================================================================
% Tested compact SSP-HBF helper functions
%% ================================================================

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


%% ================================================================
% Original direct full-matrix sensing helpers
%% ================================================================

function Q=construct_matrixQ_direct(L,noise_s,Phi,A,dAtheta,dAphi,B,dBtheta,dBphi,U)
M=size(U,1);

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
% Compute Q11

Q11 = (A * U' * (phi11 .* (dBtheta' * dBtheta)) * U * A')  + ...
    (dAtheta * U' * (phi11.* (B' * dBtheta)) * U * A')  + ...
    (A * U' * (phi11.* (dBtheta' * B)) * U * dAtheta' )  + ...
    (dAtheta * U' * (phi11.* (B' * B)) * U * dAtheta') ;

% Compute F12
Q12 = (A * U' * (2*phi12.* (dBtheta' * dBphi)) * U * A' )  + ...
    ( dAtheta * U' * (2*phi12.* (B' * dBphi)) * U * A')  + ...
    (A * U' * (2*phi12.* (dBtheta' * B)) * U * dAphi') + ...
    (dAtheta * U' * (2*phi12.* (B' * B)) * U * dAphi') ;

% Compute F13
Q13 = (A * U' * ((2*phi13+2j*phi14).* (dBtheta' * B)) * A' )  + ...
    (dAtheta * U' * ((2*phi13+2j*phi14).* (B' * B)) * A') ;

% Compute F22
Q22 = (A * U' * (phi22.* (dBphi' * dBphi)) * U * A' )  + ...
    (dAphi * U' * (phi22.* (B' * dBphi)) * U * A')  + ...
    (A * U' * (phi22.* (dBphi' * B)) * U * dAphi' )  + ...
    (dAphi * U' * (phi22.* (B' * B)) * U * dAphi' ) ;

% Compute F23
Q23 = (A * U' * ((2*phi23+2j*phi24).* (dBphi' * B)) *A')  + ...
    (dAphi * U' * ((2*phi23+2j*phi24).* (B' * B)) *A' ) ;

% Compute F33
Q33 = ( A * ((phi33+phi44+2j*phi34).* (B' * B))*A') ;

Q=2*L/noise_s*(Q11+Q12+Q13+Q22+Q23+Q33);
end


function FIM=calculateFIM_direct(L,noise_s,W,A,dAtheta,dAphi,B,dBtheta,dBphi,U)

% Inputs
% A: Matrix A
% U: Matrix U
% dAtheta: Partial derivative of A with respect to theta
% dAphi: Partial derivative of A with respect to phi
% dBtheta: Partial derivative of B with respect to theta
% dBphi: Partial derivative of B with respect to phi
% B: Matrix B
% Rx: Matrix Rx

Rx=W*W';

% Compute F11
F11 = (U * A' * Rx * A * U').' .* (dBtheta' * dBtheta) + ...
    (U * A' * Rx * dAtheta * U').' .* (B' * dBtheta) + ...
    (U * dAtheta' * Rx * A * U').' .* (dBtheta' * B) + ...
    (U * dAtheta' * Rx * dAtheta * U').' .* (B' * B);

% Compute F12
F12 = (U * A' * Rx * A * U').' .* (dBtheta' * dBphi) + ...
    (U * A' * Rx * dAtheta * U').' .* (B' * dBphi) + ...
    (U * dAphi' * Rx * A * U').' .* (dBtheta' * B) + ...
    (U * dAphi' * Rx * dAtheta * U').' .* (B' * B);

% Compute F13
F13 = (A' * Rx * A * U').' .* (dBtheta' * B) + ...
    (A' * Rx * dAtheta * U').' .* (B' * B);

% Compute F22
F22 = (U * A' * Rx * A * U').' .* (dBphi' * dBphi) + ...
    (U * A' * Rx * dAphi * U').' .* (B' * dBphi) + ...
    (U * dAphi' * Rx * A * U').' .* (dBphi' * B) + ...
    (U * dAphi' * Rx * dAphi * U').' .* (B' * B);

% Compute F23
F23 = (A' * Rx * A * U').' .* (dBphi' * B) + ...
    (A' * Rx * dAphi * U').' .* (B' * B);

% Compute F33
F33 = (A' * Rx * A).' .* (B' * B);


M=size(U,1);
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

FIM=triu(FIM)+triu(FIM)'-FIM.*eye(4*M);

FIM=2*L/noise_s*FIM;
end




