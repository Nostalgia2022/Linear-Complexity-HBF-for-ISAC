clc
clear
close all

%% ========================================================================
% FLOPs at EQUAL PERFORMANCE versus the number of transmit antennas
%
% Supervisor comment: "add one figure comparing the FLOPs of the considered
% methods for the same performance".
%
% Methods (G = 4, nominal weight, cold start, same initialization as
% Table I / Fig. 3):
%   Deep-SSP-HBF        : N_exe = 125 layers, 5/2 decomposition per layer
%   SSP-HBF             : compact matrix-chain implementation (Algorithm 2)
%   SSP-HBF (direct)    : the SAME iterates as SSP-HBF, but the FIM, the
%                         gradient and the shift are computed with explicit
%                         Nt-by-Nt matrices (Rx = W W^H, H Sigma2 H^H,
%                         A Sigma3 A^H, Nt-by-Nt eigendecomposition for the
%                         shift), i.e. the "direct implementation" of Fig. 4.
%
% Equal performance: for every channel, SSP-HBF is counted up to the first
% iteration at which it attains the objective of Deep-SSP-HBF after N_exe
% layers. If it never does within count_max iterations, all executed
% iterations are counted (conservative, favours SSP-HBF).
%
% FLOP model: real floating-point operations. One complex multiply-add
% counts as 8 FLOPs, so a complex (m x n)(n x p) product costs 8mnp.
% The numbers of backtracking trials, alternating iterations and inner
% analog updates actually executed by SSP-HBF are RECORDED in the run and
% multiplied by the per-operation counts in flops_model() at the end of
% this file. Nonlinear element-wise operations (angle/exp) count as
% c_phase FLOPs per entry, and an n x n Hermitian eigendecomposition as
% c_eig*n^3 FLOPs. Both constants can be changed below; they do not affect
% the comparison materially because they enter all methods.
%
% Channels: for Nt = 64 the fixed test channels of Table I are used; for
% other Nt, i.i.d. Rayleigh channels are drawn with a fixed seed. The
% target parameters are the same fixed test targets for all Nt.
% The transmit UPA is 8 x (Nt/8).
%
% Output: results/data_FLOPs_equal_performance_G4.mat and the figure.
%% ========================================================================

%% -------------------- User settings --------------------
Nt_list = [32 64 128 256];     % add 512 if run time allows
Nth     = 8;                   % UPA: Nth x (Nt/Nth)
G       = 4;
I_out   = 100;
N_exe   = 125;

flop_const.c_eig   = 30;       % FLOPs per n^3 for an n x n Hermitian eig
flop_const.c_phase = 10;       % FLOPs per entry for exp(j*angle(.))

use_plot_style = true;         % use plot_style('single') if available
save_eps = false;

%% -------------------- Paths --------------------
script_fullpath = mfilename('fullpath');
if isempty(script_fullpath)
    fig_dir = pwd;
else
    fig_dir = fileparts(script_fullpath);
end
project_root = fileparts(fig_dir);

dataset_file = first_existing({ ...
    fullfile(project_root,'fixed_dataset_K8_Nt64_M3.mat'), ...
    fullfile(fig_dir,'fixed_dataset_K8_Nt64_M3.mat')});
param_file = first_existing({ ...
    fullfile(project_root,'DeepSSP_depth_by_G_terminal_gap_g2max3_Fig_data.mat'), ...
    fullfile(fig_dir,'DeepSSP_depth_by_G_terminal_gap_g2max3_Fig_data.mat')});

Dset = load(dataset_file);
Plearn = load(param_file);

H_test_all = Dset.H_test_all;
alpha_test = Dset.alpha_test;
theta_test = Dset.theta_test;
phi_test   = Dset.phi_test;
Nt_dataset = size(H_test_all,2);

G_learn = double(Plearn.G_list(:).');
jG = find(G_learn==G,1);
if isempty(jG), error('G=%d not in trained parameter file.',G); end
gam = double(Plearn.gamma_B3(jG,1:N_exe));
bet = double(Plearn.beta_B3(jG,1:N_exe));
stage1_layers = double(Plearn.stage1_layers);

%% -------------------- System (as in the paper) --------------------
M = 3;  K = 8;  N_RF = K;
Nrh = 10; Nrv = 8;
Pt      = db2power(20);
noise_c = db2power(0);
noise_s = db2power(0);
L = 64;

lambda_nom = 1e3;
delta_c = 1/(1+lambda_nom);
delta_s = lambda_nom/(1+lambda_nom);

Pssp.tol = 1e-5;
Pssp.count_max = 2000;
Pssp.max_accept_trials = 20;
Pssp.eps_dec = 1e-10;
Pssp.rho_in = 2;
Pssp.mu_floor = 1e-8;
Pssp.mu_margin = 1e-8;

Pdeep.hbf_outer_iter = 5;
Pdeep.hbf_inner_iter = 2;
Pdeep.eps_dec_deep = 1e-8;
Pdeep.mu_floor = 1e-8;
Pdeep.mu_margin = 1e-8;

CM = Pssp.count_max;

out_dir = fullfile(fig_dir,'results');
if ~exist(out_dir,'dir'), mkdir(out_dir); end
save_file = fullfile(out_dir,sprintf('data_FLOPs_equal_performance_G%d.mat',G));

pool = gcp('nocreate');
if isempty(pool), pool = parpool; end
fprintf('Parallel pool: %d workers\n',pool.NumWorkers);

nN = numel(Nt_list);
Res = struct([]);

%% ========================================================================
% Main loop over Nt
% ========================================================================
for iN = 1:nN

    Nt  = Nt_list(iN);
    Ntv = Nt/Nth;
    if mod(Nt,Nth) ~= 0 || mod(Nt,G) ~= 0
        error('Nt=%d must be divisible by Nth=%d and G=%d.',Nt,Nth,G);
    end
    Z = build_groupconnected_mask(Nt,N_RF,G);

    fprintf('\n============================================================\n');
    fprintf('Nt = %d (%d x %d UPA) | G = %d | Deep N = %d\n',Nt,Nth,Ntv,G,N_exe);
    fprintf('============================================================\n');

    f0 = nan(I_out,1);
    S_obj = nan(I_out,CM); S_time = nan(I_out,CM);
    S_trials = zeros(I_out,CM); S_ao = zeros(I_out,CM); S_inner = zeros(I_out,CM);
    S_final = nan(I_out,1); S_iter = nan(I_out,1);
    D_obj = nan(I_out,N_exe+1); D_time = nan(I_out,N_exe+1);

    tN = tic;

    parfor ch = 1:I_out

        if Nt == Nt_dataset
            H = double(squeeze(H_test_all(ch,:,1:K)));
        else
            rs = RandStream('mt19937ar','Seed',100000*Nt+ch);
            H = (randn(rs,Nt,K) + 1j*randn(rs,Nt,K))/sqrt(2);
        end
        alpha = double(alpha_test(ch,1:M).');
        theta = double(theta_test(ch,1:M).');
        phi   = double(phi_test(ch,1:M).');

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

        Wa0 = Z .* exp(1j*angle(H(:,1:N_RF)));
        Wd0 = digital_update_groupconnected(Wa0,H(:,1:K));
        Wd0 = sqrt(Pt)/max(norm(Wa0*Wd0,'fro'),1e-12)*Wd0;

        [o0,s0,c0,F0] = evaluate_metrics(Pre,H,Wa0*Wd0,delta_c,delta_s,noise_c,K);

        Rs = run_ssp_counted(Pre,H,Wa0,Wd0,F0,o0,s0,c0,Nbasis,R_N,Z,Pt, ...
            delta_c,delta_s,noise_c,K,M,Pssp);
        Rd = run_deep(Pre,H,Wa0,Wd0,F0,o0,s0,c0,Nbasis,R_N,Z,Pt, ...
            delta_c,delta_s,noise_c,K,M,gam,bet,stage1_layers,Pdeep);

        f0(ch) = o0;
        S_obj(ch,:) = Rs.obj_traj;   S_time(ch,:) = Rs.time_traj;
        S_trials(ch,:) = Rs.trials;  S_ao(ch,:) = Rs.ao;  S_inner(ch,:) = Rs.inner;
        S_final(ch) = Rs.obj;        S_iter(ch) = Rs.iter;
        D_obj(ch,:) = Rd.obj_traj;   D_time(ch,:) = Rd.time_traj;
    end

    fprintf('Nt=%d finished in %.1f s wall time.\n',Nt,toc(tN));

    %% ----- FLOPs at equal objective -----
    S_nnz = nnz(Z);
    F = flops_model(Nt,K,M,N_RF,S_nnz,flop_const);

    % per-iteration FLOPs of SSP-HBF (index k = iteration k, k>=2)
    per_it_c = F.grad_c + S_trials.*(F.norm + F.eval_c) ...
             + S_ao.*F.outer_full + S_inner.*F.analog_full;
    per_it_d = F.grad_d + S_trials.*(F.norm + F.eval_d) ...
             + S_ao.*F.outer_full + S_inner.*F.analog_full;
    per_it_c(:,1) = 0;  per_it_d(:,1) = 0;      % index 1 = initial point
    per_it_c(S_trials==0) = 0;  per_it_d(S_trials==0) = 0;  % not executed

    flops_layer = F.grad_c + F.dec_fixed + F.norm + F.eval_c;
    flops_deep  = N_exe*flops_layer*ones(I_out,1);

    fD = D_obj(:,N_exe+1);
    FS = fillmissing(S_obj,'previous',2);
    TS = fillmissing(S_time,'previous',2);

    k_reach = nan(I_out,1); reached = false(I_out,1);
    flops_ssp = nan(I_out,1); flops_dir = nan(I_out,1); t_ssp = nan(I_out,1);
    for ch = 1:I_out
        k = find(FS(ch,:) >= fD(ch) - 1e-12,1,'first');
        if isempty(k)
            k = S_iter(ch) + 1;                 % all executed iterations
        else
            reached(ch) = true;
        end
        k_reach(ch) = k;
        flops_ssp(ch) = sum(per_it_c(ch,1:k));
        flops_dir(ch) = sum(per_it_d(ch,1:k));
        t_ssp(ch) = TS(ch,k);
    end

    d = max(S_final - f0,1e-12);
    q = struct();
    q.Nt = Nt; q.S = S_nnz; q.F = F;
    q.eta_deep = mean((fD - f0)./d);                     % vs full SSP ref
    q.iter_ssp_mean = mean(k_reach - 1);
    q.reached = nnz(reached);
    q.flops_deep = mean(flops_deep);
    q.flops_ssp  = mean(flops_ssp);
    q.flops_dir  = mean(flops_dir);
    q.flops_layer = flops_layer;
    q.flops_ssp_per_iter = mean(flops_ssp./max(k_reach-1,1));
    q.flops_dir_per_iter = mean(flops_dir./max(k_reach-1,1));
    q.time_deep = mean(D_time(:,N_exe+1));
    q.time_ssp  = mean(t_ssp);
    q.raw = struct('f0',f0,'fD',fD,'k_reach',k_reach,'reached',reached, ...
        'flops_ssp',flops_ssp,'flops_dir',flops_dir,'S_trials',S_trials, ...
        'S_ao',S_ao,'S_inner',S_inner,'S_obj',S_obj,'D_obj',D_obj, ...
        'S_time',S_time,'D_time',D_time,'S_final',S_final,'S_iter',S_iter);

    Res = [Res, q]; %#ok<AGROW>

    fprintf(['  Deep eta@%d = %.2f %% | SSP iters to reach = %.0f (%d/%d) | ' ...
             'GFLOPs: Deep %.3f, SSP %.3f, SSP direct %.3f | ' ...
             'ratio SSP/Deep %.1fx, direct/Deep %.1fx | time Deep %.3f s, SSP %.3f s\n'], ...
        N_exe,100*q.eta_deep,q.iter_ssp_mean,q.reached,I_out, ...
        q.flops_deep/1e9,q.flops_ssp/1e9,q.flops_dir/1e9, ...
        q.flops_ssp/q.flops_deep,q.flops_dir/q.flops_deep, ...
        q.time_deep,q.time_ssp);

    save(save_file,'Res','Nt_list','G','N_exe','flop_const','Pssp','Pdeep','I_out','-v7.3');
end

%% ========================================================================
% Summary
% ========================================================================
T = table([Res.Nt].',100*[Res.eta_deep].',[Res.iter_ssp_mean].', ...
    [Res.flops_deep].'/1e9,[Res.flops_ssp].'/1e9,[Res.flops_dir].'/1e9, ...
    ([Res.flops_ssp]./[Res.flops_deep]).',([Res.flops_dir]./[Res.flops_deep]).', ...
    'VariableNames',{'Nt','Deep_eta_pct','SSP_iters_to_reach', ...
    'GFLOPs_Deep','GFLOPs_SSP','GFLOPs_SSP_direct','SSP_over_Deep','Direct_over_Deep'});
disp(' '); disp(T);

%% ========================================================================
% Figure
% ========================================================================
S = [];
if use_plot_style && exist('plot_style','file') == 2
    S = plot_style('single');
end

fig = figure('Color','w','Name','FLOPs at equal performance');
ax = axes(fig); hold(ax,'on');

x = [Res.Nt];
h1 = loglog(ax,x,[Res.flops_dir],'-^','LineWidth',1.6,'MarkerSize',7);
h2 = loglog(ax,x,[Res.flops_ssp],'-s','LineWidth',1.6,'MarkerSize',7);
h3 = loglog(ax,x,[Res.flops_deep],'-o','LineWidth',1.6,'MarkerSize',7);
set(ax,'XScale','log','YScale','log','XTick',x);
grid(ax,'on'); box(ax,'off');
xlabel(ax,'Number of transmit antennas','Interpreter','none');
ylabel(ax,'FLOPs per channel at equal objective','Interpreter','none');
legend(ax,[h1 h2 h3],{'SSP-HBF (direct implementation)','SSP-HBF','Deep-SSP-HBF'}, ...
    'Location','northwest','Interpreter','none');

if ~isempty(S)
    set(h3,'Color',S.proposed.color); set(h2,'Color',S.baseline.color);
    plot_style('single',ax);
end

if save_eps
    print(fig,fullfile(out_dir,sprintf('Fig_FLOPs_equal_performance_G%d.eps',G)), ...
        '-depsc2','-painters');
end

save(save_file,'Res','T','Nt_list','G','N_exe','flop_const','Pssp','Pdeep','I_out','-v7.3');
fprintf('\nSaved: %s\n',save_file);

%% ========================================================================
% FLOP model
% ========================================================================
function F = flops_model(Nt,K,M,NRF,S,c)
%FLOPS_MODEL Real FLOPs of every operation used by the three methods.
%   Nt : transmit antennas, K : users, M : targets, NRF : RF chains,
%   S  : number of phase shifters nnz(Z). Complex multiply-add = 8 FLOPs.

    rN = K + 3*M;          % columns of N = [H, A, dA_theta, dA_phi]
    m3 = 3*M;

    % ---- common small terms ----
    FIM_small = 12*8*M*M*K + 60*M^2;   % Gram blocks + Hadamard assembly
    inv_FIM   = 2*(4*M)^3;              % inverse of the 4M x 4M FIM
    phi_FIM   = 2*(4*M)^3;              % Phi = F^{-1} F^{-1}
    sigma3    = 40*M^3 + 100*M^2;       % Sigma_3 assembly
    sinr      = 10*K*K;

    % ---- compact implementation (Proposition 1, Remark 1) ----
    % objective of a candidate: one product N^H W gives SINRs and Psi
    F.eval_c = 8*Nt*rN*K + FIM_small + inv_FIM + sinr;
    % gradient N(Omega1 + Omega2 N^H W) reusing N^H W, plus shift (16)
    F.grad_c = 8*rN*rN*K + 8*Nt*rN*K + 8*Nt*K ...
             + phi_FIM + sigma3 ...
             + 2*8*rN^3 + c.c_eig*rN^3;

    % ---- direct implementation (explicit Nt x Nt matrices) ----
    % Rx = W W^H, Gamma = Abar^H Rx Abar, SINRs from H^H W
    F.eval_d = 8*Nt^2*K + 8*m3*Nt^2 + 8*m3^2*Nt + 8*Nt*K*K ...
             + 60*M^2 + inv_FIM + sinr;
    % comms: (H Sigma2 H^H) W, sensing: (Abar Sigma3 Abar^H) W,
    % shift: eig of the Nt x Nt matrix N Omega2 N^H
    F.grad_d = (8*Nt^2*K + 8*Nt^2*K + 8*Nt*K) ...
             + (8*Nt*m3^2 + 8*Nt^2*m3 + 8*Nt^2*K) ...
             + phi_FIM + sigma3 ...
             + (8*Nt*rN^2 + 8*Nt^2*rN + c.c_eig*Nt^3);

    % ---- hybrid decomposition (identical for both implementations) ----
    dig   = 8*(Nt*NRF^2 - NRF^3/3) + 8*Nt*NRF*K + 4*NRF^2*K;   % LS via QR
    rwd   = 8*NRF^2*K;                                         % P P^H
    ana   = 8*S*K + 8*S*NRF + c.c_phase*S;                     % masked update
    F.outer_full  = dig + rwd + c.c_eig*NRF^3 + (8*S*K + 4*Nt*K); % + residual
    F.analog_full = ana + 4*S;                                 % + inner check
    F.dec_fixed   = 5*(dig + rwd + 2*NRF^2) + 10*ana;          % Deep 5/2
    F.norm        = 8*S*K + 4*Nt*K + 2*NRF*K;                  % power scaling
end

%% ========================================================================
% Solvers
% ========================================================================
function R = run_ssp_counted(Pre,H,Wa,Wd,FIM,obj0,sr0,crlb0,Nbasis,R_N,Z,Pt, ...
    delta_c,delta_s,noise_c,K,M,P)
%RUN_SSP_COUNTED SSP-HBF identical to Table I, recording per iteration the
% number of candidates (trials), alternating iterations and inner analog
% updates actually executed (summed over all trials of that iteration).

    obj_traj  = nan(1,P.count_max);
    time_traj = nan(1,P.count_max);
    trials_tr = zeros(1,P.count_max);
    ao_tr     = zeros(1,P.count_max);
    inner_tr  = zeros(1,P.count_max);
    obj_traj(1) = obj0;
    time_traj(1) = 0;

    W = Wa*Wd;
    Con_prev = [obj0, crlb0, sr0];
    Con_new = Con_prev;
    last_iter = 1;
    fail = false;

    t0 = tic;

    for iter = 2:P.count_max

        W_prev = W; Wa_prev = Wa; Wd_prev = Wd;
        obj_prev = Con_prev(1);

        [grad_total,mu] = compute_grad_and_local_curvature( ...
            Pre,H,W_prev,FIM,Nbasis,R_N,delta_c,delta_s, ...
            noise_c,K,M,P.mu_floor,P.mu_margin);

        accepted = false;
        trial = 0;
        ao_sum = 0; inner_sum = 0;

        while ~accepted && trial < P.max_accept_trials
            trial = trial + 1;

            W_target = grad_total + mu*W_prev;

            [~,Wa_c,Wd_c,ao_used,inner_used] = ...
                matrix_approximation_groupconnected_counted( ...
                W_target,Wa_prev,Wd_prev,Z);
            ao_sum = ao_sum + ao_used;
            inner_sum = inner_sum + inner_used;

            Wd_c = sqrt(Pt)/max(norm(Wa_c*Wd_c,'fro'),1e-12)*Wd_c;
            W_c = Wa_c*Wd_c;

            [o_c,s_c,c_c,F_c] = evaluate_metrics( ...
                Pre,H,W_c,delta_c,delta_s,noise_c,K);

            if o_c >= obj_prev - P.eps_dec
                accepted = true;
                W = W_c; Wa = Wa_c; Wd = Wd_c; FIM = F_c;
                Con_new = [o_c, c_c, s_c];
                last_iter = iter;
            else
                mu = P.rho_in*mu;
            end
        end

        trials_tr(iter) = trial;
        ao_tr(iter) = ao_sum;
        inner_tr(iter) = inner_sum;

        if ~accepted
            fail = true;
            break;
        end

        obj_traj(iter) = Con_new(1);
        time_traj(iter) = toc(t0);

        relchg = abs(Con_new-Con_prev)./max(abs(Con_prev),1e-12);
        Con_prev = Con_new;

        if all(relchg <= P.tol)
            break;
        end
    end

    R.time = toc(t0);
    R.obj  = Con_prev(1);
    R.crlb = Con_prev(2);
    R.sr   = Con_prev(3);
    R.iter = last_iter - 1;
    R.fail = fail;
    R.obj_traj = obj_traj;
    R.time_traj = time_traj;
    R.trials = trials_tr;
    R.ao = ao_tr;
    R.inner = inner_tr;
end


function [obj, Wa, Wd, iter_used, inner_total] = ...
    matrix_approximation_groupconnected_counted(W, Wa, Wd, Z)
%MATRIX_APPROXIMATION_GROUPCONNECTED_COUNTED Identical to
% matrix_approximation_groupconnected, additionally returning the total
% number of inner analog updates.

    count_max  = 200;
    inner_iter = 20;
    tol_outer  = 1e-5;
    tol_inner  = 1e-4;

    obj = zeros(count_max, 1);
    Wa = Z .* exp(1j * angle(Wa));
    inner_total = 0;

    for count = 1:count_max

        Wd = digital_update_groupconnected(Wa, W);

        Rwd = Wd * Wd';
        Rwd = (Rwd + Rwd') / 2;
        mu1 = max(real(eig(Rwd)));

        for iter = 1:inner_iter
            Wa_old = Wa;
            D = W * Wd' + Wa * (mu1 * eye(size(Rwd)) - Rwd);
            Wa = Z .* exp(1j * angle(D));
            inner_total = inner_total + 1;
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


function R = run_deep(Pre,H,Wa,Wd,FIM,obj0,sr0,crlb0,Nbasis,R_N,Z,Pt, ...
    delta_c,delta_s,noise_c,K,M,gam,bet,stage1_layers,P)
%RUN_DEEP Frozen Deep-SSP-HBF, identical to the Deep tradeoff script.
% Trajectory index t+1 = after layer t (index 1: initial).

    N = numel(gam);
    obj_traj  = nan(1,N+1);
    time_traj = nan(1,N+1);
    obj_traj(1) = obj0;
    time_traj(1) = 0;

    W = Wa*Wd;
    W_prev_layer = W;
    obj_cur = obj0; sr_cur = sr0; crlb_cur = crlb0;
    rej = 0;

    t0 = tic;

    for t = 1:N

        W_old = W; Wa_old = Wa; Wd_old = Wd;
        obj_old = obj_cur;

        [grad_total,lambda_local] = compute_grad_and_local_curvature( ...
            Pre,H,W_old,FIM,Nbasis,R_N,delta_c,delta_s, ...
            noise_c,K,M,P.mu_floor,P.mu_margin);

        if t <= stage1_layers
            beta_t = 0;
        else
            beta_t = bet(t);
        end

        inertial_state = W_old + beta_t*(W_old-W_prev_layer);
        W_target = grad_total + ...
            (lambda_local/max(gam(t),1e-8))*inertial_state;

        [Wa_c,Wd_c] = matrix_approximation_groupconnected_fixed( ...
            W_target,Wa_old,Wd_old,Z,P.hbf_outer_iter,P.hbf_inner_iter);

        Wd_c = sqrt(Pt)/max(norm(Wa_c*Wd_c,'fro'),1e-12)*Wd_c;
        W_c = Wa_c*Wd_c;

        [o_c,s_c,c_c,F_c] = evaluate_metrics( ...
            Pre,H,W_c,delta_c,delta_s,noise_c,K);

        W_prev_layer = W_old;

        if o_c >= obj_old - P.eps_dec_deep
            W = W_c; Wa = Wa_c; Wd = Wd_c; FIM = F_c;
            obj_cur = o_c; sr_cur = s_c; crlb_cur = c_c;
        else
            rej = rej + 1;      % keep current iterate
        end

        obj_traj(t+1) = obj_cur;
        time_traj(t+1) = toc(t0);
    end

    R.time = toc(t0);
    R.obj = obj_cur;
    R.sr = sr_cur;
    R.crlb = crlb_cur;
    R.rej = rej;
    R.obj_traj = obj_traj;
    R.time_traj = time_traj;
end


function f = first_existing(c)
    for k = 1:numel(c)
        if exist(c{k},'file')
            f = c{k};
            return;
        end
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
