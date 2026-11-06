clc
clear
close all

%% ========================================================================
% TABLE I / CONVERGENCE-FIGURE DATA (v2): nominal point, COLD START
%
% Compares, on the same 100 test channels, from the same initialization,
% at lambda = delta_s/delta_c = 1e3, for G = 1, 2, 4, 8:
%
%   SSP-HBF (full) : decomposition as in Algorithm 1 of the paper
%                    (I_AO = 200, I_in = 20, exact lambda_max shift,
%                     tolerances 1e-5 / 1e-4)
%   SSP-HBF (5/2)  : SAME SSP-HBF outer iteration and backtracking, but the
%                    decomposition uses the Deep-SSP-HBF budget
%                    (I_AO = 5, I_in = 2, Gershgorin shift)
%   Deep-SSP-HBF   : frozen model, executed to N_run = 200 layers;
%                    Table I values are read at N_exe = 125 from the same
%                    trajectory, so the convergence figure (0-200 layers)
%                    and Table I come from ONE run.
%
% Purpose: check whether the existing convergence figure
%   (data_DeepSSP_runtime_accuracy_depth_G4_100ch.mat) used the 5/2
%   decomposition for SSP-HBF, and provide consistent data for both the
%   convergence figure and Table I.
%
% Normalized progress is reported with TWO references:
%   ref = 'full' : f_ref = SSP-HBF (full) after count_max iterations
%   ref = '52'   : f_ref = SSP-HBF (5/2)  after count_max iterations
% and with two channel averages:
%   MoR : mean over channels of (f - f0)/(fref - f0)   (as in the figure)
%   RoM : (mean f - mean f0)/(mean fref - mean f0)
%
% Timing: clock starts after the common setup/initialization; all three
% methods run back-to-back in the same single-threaded worker, in a
% channel-dependent rotating order.
%% ========================================================================

%% ------------------------------------------------------------------------
% Paths
% -------------------------------------------------------------------------
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

G_learn = double(Plearn.G_list(:).');
gamma_B3_all = double(Plearn.gamma_B3);
beta_B3_all  = double(Plearn.beta_B3);
stage1_layers = double(Plearn.stage1_layers);

fprintf('Dataset    : %s\n',dataset_file);
fprintf('Parameters : %s\n',param_file);

%% ------------------------------------------------------------------------
% System setting (identical to the tradeoff scripts)
% -------------------------------------------------------------------------
M = 3;
K = 8;
Nth = 8;  Ntv = 8;  Nt = Nth*Ntv;
Nrh = 10; Nrv = 8;  Nr = Nrh*Nrv;
N_RF = K;
G_list = [1 2 4 8];

Pt      = db2power(20);
noise_c = db2power(0);
noise_s = db2power(0);
L = 64;

I_out = 100;

lambda_nom = 1e3;
delta_c = 1/(1+lambda_nom);
delta_s = lambda_nom/(1+lambda_nom);

%% ------------------------------------------------------------------------
% Algorithm parameters
% -------------------------------------------------------------------------
Pssp.tol = 1e-5;
Pssp.count_max = 2000;
Pssp.max_accept_trials = 20;
Pssp.eps_dec = 1e-10;
Pssp.rho_in = 2;
Pssp.mu_floor = 1e-8;
Pssp.mu_margin = 1e-8;
Pssp.use_stopping_test = true;   % set false to force all count_max iterations

Pssp_full = Pssp;  Pssp_full.decomp = 'full';
Pssp_52   = Pssp;  Pssp_52.decomp   = 'fixed';
Pssp_52.hbf_outer_iter = 5;
Pssp_52.hbf_inner_iter = 2;

N_run = 200;          % Deep layers executed (convergence figure)
N_exe = 125;          % Deep depth reported in Table I
Pdeep.hbf_outer_iter = 5;
Pdeep.hbf_inner_iter = 2;
Pdeep.eps_dec_deep = 1e-8;
Pdeep.mu_floor = 1e-8;
Pdeep.mu_margin = 1e-8;

threshold = 0.99;

%% ------------------------------------------------------------------------
% Output
% -------------------------------------------------------------------------
out_dir = fullfile(fig_dir,'results');
if ~exist(out_dir,'dir'), mkdir(out_dir); end

save_file = fullfile(out_dir, ...
    sprintf('data_TableI_v2_full_vs_52_N%d_run%d_100ch.mat',N_exe,N_run));

pool = gcp('nocreate');
if isempty(pool), pool = parpool; end
fprintf('Parallel pool: %d workers\n',pool.NumWorkers);

nG = numel(G_list);
Res = struct([]);
CM = Pssp.count_max;

%% ========================================================================
% Main loop
% ========================================================================
for iG = 1:nG

    G = G_list(iG);
    Z = build_groupconnected_mask(Nt,N_RF,G);

    jG = find(G_learn==G,1);
    if isempty(jG), error('G=%d not in trained parameter file.',G); end
    if size(gamma_B3_all,2) < N_run
        error('Trained parameters have fewer than %d layers.',N_run);
    end
    gam = gamma_B3_all(jG,1:N_run);
    bet = beta_B3_all(jG,1:N_run);

    fprintf('\n============================================================\n');
    fprintf('G=%d | lambda=%g | cold start | SSP full, SSP 5/2, Deep N=%d\n', ...
        G,lambda_nom,N_run);
    fprintf('============================================================\n');

    f0 = nan(I_out,1);

    A_obj = nan(I_out,1); A_sr = nan(I_out,1); A_crlb = nan(I_out,1);
    A_time = nan(I_out,1); A_iter = nan(I_out,1); A_fail = false(I_out,1);
    A_ot = nan(I_out,CM); A_tt = nan(I_out,CM); A_bt = nan(I_out,CM);

    B_obj = nan(I_out,1); B_sr = nan(I_out,1); B_crlb = nan(I_out,1);
    B_time = nan(I_out,1); B_iter = nan(I_out,1); B_fail = false(I_out,1);
    B_ot = nan(I_out,CM); B_tt = nan(I_out,CM); B_bt = nan(I_out,CM);

    D_ot = nan(I_out,N_run+1); D_tt = nan(I_out,N_run+1);
    D_srt = nan(I_out,N_run+1); D_crt = nan(I_out,N_run+1);
    D_rej = nan(I_out,N_run+1);

    tG = tic;

    parfor ch = 1:I_out

        H = double(squeeze(H_test_all(ch,:,1:K)));
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

        % rotate execution order over channels: 1-2-3, 2-3-1, 3-1-2
        order = circshift(1:3,-mod(ch-1,3));
        Ra = []; Rb = []; Rd = [];
        for m = order
            switch m
                case 1
                    Ra = run_ssp(Pre,H,Wa0,Wd0,F0,o0,s0,c0,Nbasis,R_N,Z,Pt, ...
                        delta_c,delta_s,noise_c,K,M,Pssp_full);
                case 2
                    Rb = run_ssp(Pre,H,Wa0,Wd0,F0,o0,s0,c0,Nbasis,R_N,Z,Pt, ...
                        delta_c,delta_s,noise_c,K,M,Pssp_52);
                case 3
                    Rd = run_deep(Pre,H,Wa0,Wd0,F0,o0,s0,c0,Nbasis,R_N,Z,Pt, ...
                        delta_c,delta_s,noise_c,K,M,gam,bet,stage1_layers,Pdeep);
            end
        end

        f0(ch) = o0;

        A_obj(ch) = Ra.obj; A_sr(ch) = Ra.sr; A_crlb(ch) = Ra.crlb;
        A_time(ch) = Ra.time; A_iter(ch) = Ra.iter; A_fail(ch) = Ra.fail;
        A_ot(ch,:) = Ra.obj_traj; A_tt(ch,:) = Ra.time_traj; A_bt(ch,:) = Ra.bt_traj;

        B_obj(ch) = Rb.obj; B_sr(ch) = Rb.sr; B_crlb(ch) = Rb.crlb;
        B_time(ch) = Rb.time; B_iter(ch) = Rb.iter; B_fail(ch) = Rb.fail;
        B_ot(ch,:) = Rb.obj_traj; B_tt(ch,:) = Rb.time_traj; B_bt(ch,:) = Rb.bt_traj;

        D_ot(ch,:) = Rd.obj_traj; D_tt(ch,:) = Rd.time_traj;
        D_srt(ch,:) = Rd.sr_traj; D_crt(ch,:) = Rd.crlb_traj;
        D_rej(ch,:) = Rd.rej_traj;
    end

    fprintf('G=%d finished in %.1f s wall time.\n',G,toc(tG));

    %% ----- store raw data -----
    R = struct();
    R.G = G;
    R.f0 = f0;
    R.full = struct('obj',A_obj,'sr',A_sr,'crlb',A_crlb,'time',A_time, ...
        'iter',A_iter,'fail',A_fail,'obj_traj',A_ot,'time_traj',A_tt,'bt_traj',A_bt);
    R.d52  = struct('obj',B_obj,'sr',B_sr,'crlb',B_crlb,'time',B_time, ...
        'iter',B_iter,'fail',B_fail,'obj_traj',B_ot,'time_traj',B_tt,'bt_traj',B_bt);
    R.deep = struct('obj_traj',D_ot,'time_traj',D_tt,'sr_traj',D_srt, ...
        'crlb_traj',D_crt,'rej_traj',D_rej);

    %% ----- summaries for both references -----
    R.sum_ref_full = summarize(R,R.full.obj,N_exe,threshold);
    R.sum_ref_52   = summarize(R,R.d52.obj ,N_exe,threshold);

    Res = [Res, R]; %#ok<AGROW>

    print_summary(R,N_exe,N_run);

    save(save_file,'Res','G_list','lambda_nom','delta_c','delta_s', ...
        'N_exe','N_run','Pssp_full','Pssp_52','Pdeep','I_out','threshold','-v7.3');
end

%% ========================================================================
% Compact overview + LaTeX table (reference = SSP-HBF full, MoR)
% ========================================================================
fprintf('\n=========================================================================\n');
fprintf('OVERVIEW (MoR, ref = SSP-HBF full after %d iterations)\n',CM-1);
fprintf('=========================================================================\n');
fprintf(['  G | ms/iter full | ms/iter 5/2 | ms/layer Deep | Deep eta@%d | ' ...
         'Deep N99 | full N99 | 5/2 N99 | 5/2 eta@2000\n'],N_exe);
for iG = 1:nG
    s = Res(iG).sum_ref_full;
    fprintf('  %d | %12.3f | %11.3f | %13.3f | %9.2f %% | %8s | %8s | %7s | %8.2f %%\n', ...
        Res(iG).G,s.ms_per_iter_full,s.ms_per_iter_52,s.ms_per_layer_deep, ...
        100*s.MoR.deep_eta,num2s(s.MoR.deep_N99),num2s(s.MoR.full_N99), ...
        num2s(s.MoR.d52_N99),100*s.MoR.d52_eta_end);
end

fprintf('\nSame overview with ref = SSP-HBF (5/2) after %d iterations (MoR):\n',CM-1);
for iG = 1:nG
    s = Res(iG).sum_ref_52;
    fprintf('  G=%d | Deep eta@%d %.2f %% | Deep N99 %s (%.4f s) | 5/2 N99 %s (%.4f s) | speedup@99 %.2fx\n', ...
        Res(iG).G,N_exe,100*s.MoR.deep_eta,num2s(s.MoR.deep_N99),s.MoR.deep_T99, ...
        num2s(s.MoR.d52_N99),s.MoR.d52_T99,s.MoR.d52_T99/s.MoR.deep_T99);
end

% LaTeX Table I (ref = SSP-HBF full, MoR, SSP-HBF = full decomposition)
Ltex = {};
Ltex{end+1} = '\begin{table}[t]';
Ltex{end+1} = '\centering';
Ltex{end+1} = ['\caption{Achieved fraction $\eta$ of the reference objective ' ...
    'improvement and per-channel CPU time at the nominal operating point, ' ...
    'with both methods started from the same initialization.}'];
Ltex{end+1} = '\label{tab:runtime}';
Ltex{end+1} = '\setlength{\tabcolsep}{4pt}';
Ltex{end+1} = '\begin{tabular}{c ccc cc c}';
Ltex{end+1} = '\hline';
Ltex{end+1} = '& \multicolumn{3}{c}{SSP-HBF} & \multicolumn{2}{c}{Deep-SSP-HBF} & \\';
Ltex{end+1} = '\cline{2-4}\cline{5-6}';
Ltex{end+1} = '& $\eta$ (\%) & \multicolumn{2}{c}{To reach Deep} & $\eta$ (\%) & Time & \\';
Ltex{end+1} = sprintf('$G$ & $%d$ iter. & Iter. & Time (s) & $N_{\\rm exe}{=}%d$ & (s) & Speedup \\\\',N_exe,N_exe);
Ltex{end+1} = '\hline';
for iG = 1:nG
    s = Res(iG).sum_ref_full;
    Ltex{end+1} = sprintf('%d & %.1f & %.0f & %.3f & %.1f & %.3f & $%.1f\\times$ \\\\', ...
        Res(iG).G,100*s.MoR.full_eta_Nexe,s.iso_iter,s.iso_time, ...
        100*s.MoR.deep_eta,s.deep_time_Nexe,s.iso_speedup); %#ok<SAGROW>
end
Ltex{end+1} = '\hline';
Ltex{end+1} = '\end{tabular}';
Ltex{end+1} = '\end{table}';

fprintf('\n'); fprintf('%s\n',Ltex{:});

tex_file = fullfile(out_dir,'TableI_v2_nominal_coldstart.tex');
fid = fopen(tex_file,'w'); fprintf(fid,'%s\n',Ltex{:}); fclose(fid);

save(save_file,'Res','G_list','lambda_nom','delta_c','delta_s', ...
    'N_exe','N_run','Pssp_full','Pssp_52','Pdeep','I_out','threshold','-v7.3');

fprintf('\nSaved:\n  %s\n  %s\n',save_file,tex_file);

%% ========================================================================
% Optional: compare with the existing convergence-figure file (G = 4)
% ========================================================================
fig_name = 'data_DeepSSP_runtime_accuracy_depth_G4_100ch.mat';
hit = dir(fullfile(project_root,'**',fig_name));
iG4 = find([Res.G]==4,1);

if ~isempty(hit) && ~isempty(iG4)
    Dfig = load(fullfile(hit(1).folder,hit(1).name));
    r = Res(iG4);
    fprintf('\n============================================================\n');
    fprintf('Comparison with existing convergence-figure file (G = 4)\n');
    fprintf('============================================================\n');
    f0_fig = double(Dfig.Obj_B1_all(:,1));
    fprintf('  max |f0_fig - f0_new|                   = %.3e\n', ...
        max(abs(f0_fig - r.f0)));
    fprintf('  max |RefObj_fig - SSP(full) final|      = %.3e\n', ...
        max(abs(double(Dfig.RefObj_all) - r.full.obj)));
    fprintf('  max |RefObj_fig - SSP(5/2) final|       = %.3e\n', ...
        max(abs(double(Dfig.RefObj_all) - r.d52.obj)));
    fprintf('  max |Obj_B1_fig(end) - SSP(full) final| = %.3e\n', ...
        max(abs(double(Dfig.Obj_B1_all(:,end)) - r.full.obj)));
    fprintf('  max |Obj_B1_fig(end) - SSP(5/2) final|  = %.3e\n', ...
        max(abs(double(Dfig.Obj_B1_all(:,end)) - r.d52.obj)));
    fprintf('  hbf_outer/inner stored in figure file   = %g / %g\n', ...
        Dfig.hbf_outer_iter,Dfig.hbf_inner_iter);
    fprintf('  ms/iter: figure B1 %.3f | new full %.3f | new 5/2 %.3f\n', ...
        1e3*Dfig.Time_B1_avg(end)/(numel(Dfig.Time_B1_avg)-1), ...
        r.sum_ref_full.ms_per_iter_full,r.sum_ref_full.ms_per_iter_52);

    % overlay: figure curves vs new curves (MoR, figure's own reference)
    pf_fig_ssp  = 1 - double(Dfig.Gap_B1_avg(:).');
    pf_fig_deep = 1 - double(Dfig.Gap_Deep_avg(:).');
    s52 = r.sum_ref_52.MoR; sfu = r.sum_ref_full.MoR;

    figure('Name','Figure file vs new run (G = 4)','Color','w');
    subplot(1,2,1); hold on; grid on;
    plot(0:numel(pf_fig_deep)-1,pf_fig_deep,'k-','LineWidth',1.5);
    plot(0:N_run,sfu.deep_curve,'b--','LineWidth',1.2);
    plot(0:N_run,s52.deep_curve,'r-.','LineWidth',1.2);
    yline(threshold,'k:'); xline(N_exe,'k:');
    xlim([0 N_run+5]); ylim([0.88 1.01]);
    xlabel('Layer'); ylabel('Normalized progress'); title('Deep-SSP-HBF');
    legend({'Figure file','New, ref = full','New, ref = 5/2'},'Location','southeast');

    subplot(1,2,2); hold on; grid on;
    plot(0:numel(pf_fig_ssp)-1,pf_fig_ssp,'k-','LineWidth',1.5);
    plot(0:CM-1,sfu.full_curve,'b--','LineWidth',1.2);
    plot(0:CM-1,s52.d52_curve,'r-.','LineWidth',1.2);
    yline(threshold,'k:');
    ylim([0.8 1.01]);
    xlabel('Iteration'); ylabel('Normalized progress'); title('SSP-HBF');
    legend({'Figure file','New SSP full (ref full)','New SSP 5/2 (ref 5/2)'}, ...
        'Location','southeast');
end

%% ========================================================================
% Local functions: solvers
% ========================================================================
function R = run_ssp(Pre,H,Wa,Wd,FIM,obj0,sr0,crlb0,Nbasis,R_N,Z,Pt, ...
    delta_c,delta_s,noise_c,K,M,P)
%RUN_SSP Cold-start SSP-HBF. P.decomp = 'full' (Algorithm 1, I_AO=200,
% I_in=20) or 'fixed' (I_AO=P.hbf_outer_iter, I_in=P.hbf_inner_iter,
% Gershgorin shift, as in Deep-SSP-HBF).
% Trajectory index k = after k-1 accepted updates (k = 1: initial point).

    obj_traj  = nan(1,P.count_max);
    time_traj = nan(1,P.count_max);
    bt_traj   = nan(1,P.count_max);
    obj_traj(1) = obj0;
    time_traj(1) = 0;
    bt_traj(1) = 0;

    W = Wa*Wd;
    Con_prev = [obj0, crlb0, sr0];
    Con_new = Con_prev;
    last_iter = 1;
    fail = false;
    use_fixed = strcmp(P.decomp,'fixed');

    t0 = tic;

    for iter = 2:P.count_max

        W_prev = W; Wa_prev = Wa; Wd_prev = Wd;
        obj_prev = Con_prev(1);

        [grad_total,mu] = compute_grad_and_local_curvature( ...
            Pre,H,W_prev,FIM,Nbasis,R_N,delta_c,delta_s, ...
            noise_c,K,M,P.mu_floor,P.mu_margin);

        accepted = false;
        trial = 0;

        while ~accepted && trial < P.max_accept_trials
            trial = trial + 1;

            W_target = grad_total + mu*W_prev;

            if use_fixed
                [Wa_c,Wd_c] = matrix_approximation_groupconnected_fixed( ...
                    W_target,Wa_prev,Wd_prev,Z,P.hbf_outer_iter,P.hbf_inner_iter);
            else
                [~,Wa_c,Wd_c] = matrix_approximation_groupconnected( ...
                    W_target,Wa_prev,Wd_prev,Z);
            end

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

        if ~accepted
            fail = true;
            break;
        end

        obj_traj(iter) = Con_new(1);
        time_traj(iter) = toc(t0);
        bt_traj(iter) = trial - 1;

        relchg = abs(Con_new-Con_prev)./max(abs(Con_prev),1e-12);
        Con_prev = Con_new;

        if P.use_stopping_test && all(relchg <= P.tol)
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
    R.bt_traj = bt_traj;
end


function R = run_deep(Pre,H,Wa,Wd,FIM,obj0,sr0,crlb0,Nbasis,R_N,Z,Pt, ...
    delta_c,delta_s,noise_c,K,M,gam,bet,stage1_layers,P)
%RUN_DEEP Frozen Deep-SSP-HBF (identical to the Deep tradeoff script).
% Trajectory index t+1 = after layer t (index 1: initial point).

    N = numel(gam);
    obj_traj  = nan(1,N+1); time_traj = nan(1,N+1);
    sr_traj   = nan(1,N+1); crlb_traj = nan(1,N+1);
    rej_traj  = nan(1,N+1);
    obj_traj(1) = obj0; sr_traj(1) = sr0; crlb_traj(1) = crlb0;
    time_traj(1) = 0; rej_traj(1) = 0;

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
            rej = rej + 1;
        end

        obj_traj(t+1) = obj_cur; sr_traj(t+1) = sr_cur;
        crlb_traj(t+1) = crlb_cur; rej_traj(t+1) = rej;
        time_traj(t+1) = toc(t0);
    end

    R.obj_traj = obj_traj; R.time_traj = time_traj;
    R.sr_traj = sr_traj; R.crlb_traj = crlb_traj; R.rej_traj = rej_traj;
end

%% ========================================================================
% Local functions: post-processing
% ========================================================================
function s = summarize(R,fref,N_exe,thr)
%SUMMARIZE Progress curves, 99% points and Table I quantities for a given
% per-channel reference fref.

    f0 = R.f0(:);
    fref = fref(:);
    d = max(fref - f0,1e-12);

    FA = fillmissing(R.full.obj_traj,'previous',2);
    FB = fillmissing(R.d52.obj_traj ,'previous',2);
    FD = R.deep.obj_traj;
    TA = fillmissing(R.full.time_traj,'previous',2);
    TB = fillmissing(R.d52.time_traj ,'previous',2);
    TD = R.deep.time_traj;

    tA = mean(TA,1); tB = mean(TB,1); tD = mean(TD,1);

    for typ = {'MoR','RoM'}
        if strcmp(typ{1},'MoR')
            cA = mean((FA - f0)./d,1);
            cB = mean((FB - f0)./d,1);
            cD = mean((FD - f0)./d,1);
        else
            cA = (mean(FA,1) - mean(f0))/mean(d);
            cB = (mean(FB,1) - mean(f0))/mean(d);
            cD = (mean(FD,1) - mean(f0))/mean(d);
        end
        q.full_curve = cA; q.d52_curve = cB; q.deep_curve = cD;
        [q.full_N99,q.full_T99] = first_cross(cA,tA,thr);
        [q.d52_N99 ,q.d52_T99 ] = first_cross(cB,tB,thr);
        [q.deep_N99,q.deep_T99] = first_cross(cD,tD,thr);
        q.deep_eta = cD(N_exe+1);
        q.deep_eta_end = cD(end);
        q.full_eta_Nexe = cA(min(N_exe+1,end));
        q.d52_eta_Nexe  = cB(min(N_exe+1,end));
        q.full_eta_end = cA(end);
        q.d52_eta_end  = cB(end);
        s.(typ{1}) = q;
    end

    % iso-quality: SSP-HBF (full) time/iterations to reach Deep at N_exe
    fD = FD(:,N_exe+1);
    nC = numel(f0);
    it = nan(nC,1); tt = nan(nC,1); reached = false(nC,1);
    for ch = 1:nC
        k = find(FA(ch,:) >= fD(ch) - 1e-12,1,'first');
        if isempty(k)
            it(ch) = R.full.iter(ch); tt(ch) = R.full.time(ch);
        else
            it(ch) = k-1; tt(ch) = TA(ch,k); reached(ch) = true;
        end
    end
    s.deep_time_Nexe = mean(TD(:,N_exe+1));
    s.iso_iter = mean(it);
    s.iso_time = mean(tt);
    s.iso_reached = nnz(reached);
    s.iso_speedup = s.iso_time/s.deep_time_Nexe;

    s.ms_per_iter_full  = 1e3*mean(R.full.time./max(R.full.iter,1));
    s.ms_per_iter_52    = 1e3*mean(R.d52.time ./max(R.d52.iter ,1));
    s.ms_per_layer_deep = 1e3*tD(end)/(size(TD,2)-1);
    s.mean_bt_full = mean(R.full.bt_traj(:,2:end),'all','omitnan');
    s.mean_bt_52   = mean(R.d52.bt_traj(:,2:end),'all','omitnan');
end


function print_summary(R,N_exe,N_run)
    s = R.sum_ref_full; s2 = R.sum_ref_52;
    fprintf('  SSP full : %.1f upd, %.3f s, %.3f ms/upd, bt %.2f | SR %.3f | CRLB %.4e | fail %d\n', ...
        mean(R.full.iter),mean(R.full.time),s.ms_per_iter_full,s.mean_bt_full, ...
        mean(R.full.sr),mean(R.full.crlb),nnz(R.full.fail));
    fprintf('  SSP 5/2  : %.1f upd, %.3f s, %.3f ms/upd, bt %.2f | SR %.3f | CRLB %.4e | fail %d\n', ...
        mean(R.d52.iter),mean(R.d52.time),s.ms_per_iter_52,s.mean_bt_52, ...
        mean(R.d52.sr),mean(R.d52.crlb),nnz(R.d52.fail));
    fprintf('  Deep     : %.3f ms/layer | t(%d) %.4f s | t(%d) %.4f s | rejects@%d %.2f\n', ...
        s.ms_per_layer_deep,N_exe,s.deep_time_Nexe,N_run, ...
        mean(R.deep.time_traj(:,end)),N_exe,mean(R.deep.rej_traj(:,N_exe+1)));
    fprintf('  mean f_ref: full %.6e | 5/2 %.6e | (5/2 - full)/(full - f0) = %+.3f %%\n', ...
        mean(R.full.obj),mean(R.d52.obj), ...
        100*mean(R.d52.obj - R.full.obj)/mean(R.full.obj - R.f0));
    for pair = {{'ref full',s},{'ref 5/2',s2}}
        nm = pair{1}{1}; q = pair{1}{2};
        for typ = {'MoR','RoM'}
            z = q.(typ{1});
            fprintf(['  [%s, %s] Deep eta@%d %.2f %% (@%d %.2f %%), N99 %s (%s s) | ' ...
                     'full N99 %s (%s s) | 5/2 N99 %s (%s s) | eta@%d full %.2f %%, 5/2 %.2f %%\n'], ...
                nm,typ{1},N_exe,100*z.deep_eta,N_run,100*z.deep_eta_end, ...
                num2s(z.deep_N99),num2s(z.deep_T99,'%.4f'), ...
                num2s(z.full_N99),num2s(z.full_T99,'%.4f'), ...
                num2s(z.d52_N99),num2s(z.d52_T99,'%.4f'), ...
                N_exe,100*z.full_eta_Nexe,100*z.d52_eta_Nexe);
        end
    end
    fprintf('  Table I (ref full): SSP full to reach Deep@%d: %.1f iter, %.4f s (%d/100) | speedup %.2fx\n', ...
        N_exe,s.iso_iter,s.iso_time,s.iso_reached,s.iso_speedup);
end


function [n,t] = first_cross(p,tt,thr)
    k = find(p >= thr,1,'first');
    if isempty(k), n = []; t = nan; else, n = k-1; t = tt(k); end
end


function s = num2s(x,fmt)
    if nargin < 2, fmt = '%d'; end
    if isempty(x) || (isnumeric(x) && all(isnan(x))), s = '--'; else, s = sprintf(fmt,x); end
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
