clc
clear
close all

%% ========================================================================
% DIAGNOSTIC: suspicious Pt = 30 dBm SSP-HBF reference point
%
% Purpose
% -------
% Determine whether the unusual 30-dBm SSP-HBF point is caused mainly by
%
%   (A) an insufficient iteration budget / very slow convergence, or
%   (B) convergence to a poorer stationary point.
%
% The diagnostic uses EXACTLY the same:
%   * 100-channel fixed Pt-sweep dataset,
%   * G = 4 group-connected architecture,
%   * objective weights,
%   * local-curvature SSP update,
%   * HBF projection,
%   * learned Deep-SSP schedule
%
% as HBF_DeepSSP_Fig5_generalization_Pt_Nt_G4_100ch_v2.m.
%
% For the channels where Deep-SSP N=125 most strongly outperforms the
% saved long-run SSP reference at Pt=30 dBm, this script performs:
%
%   1) DEFAULT SSP-30:
%      Re-run conventional SSP-HBF from the original initialization for
%      substantially more than 2000 iterations.
%
%   2) DEEP125 -> SSP:
%      Reconstruct the Deep-SSP-HBF N=125 solution at 30 dBm and then
%      continue with the ordinary SSP-HBF update.
%
%   3) OPTIONAL Pt=25 -> Pt=30 continuation:
%      First solve SSP-HBF at 25 dBm, rescale the converged state to
%      30 dBm, and then continue SSP-HBF at 30 dBm.
%
% Interpretation
% --------------
% * If DEFAULT SSP keeps improving well after 2000 iterations and reaches
%   the Deep-refined solution, the main issue is the iteration budget.
%
% * If DEFAULT SSP is essentially flat but DEEP125 -> SSP remains at a
%   clearly better objective, the original SSP reference is in a poorer
%   stationary basin.
%
% * If the reconstructed Deep125 result does not match the saved Deep125
%   result, first check data/model consistency before drawing conclusions.
%
% Output
% ------
% results/diagnostic_Pt30_SSP_reference.mat
%
% Two diagnostic figures:
%   (1) mean objective gain over the saved SSP reference versus iteration
%   (2) average SR-CRLB operating points for the tested solutions
%% ========================================================================

%% -------------------- User options --------------------------------------
target_Pt_dBm = 30;

% Number of suspicious channels to diagnose.
% The script ranks all channels by
%   Deep125 objective - saved SSP-reference objective.
n_diag_channels = 8;

% Set a nonempty vector to override automatic channel selection, e.g.
% manual_channels = [3 17 44];
manual_channels = [];

% Extended conventional SSP run at 30 dBm.
% Must be > 2000 if the goal is to test whether 2000 iterations are enough.
ssp30_max_iter = 5000;

% Ordinary SSP refinement initialized from the Deep N=125 solution.
deep_refine_iter = 2000;

% Optional continuation test from 25 dBm.
do_pt25_continuation = false;
pt25_ssp_iter = 2000;
pt25_to_30_refine_iter = 2000;

% Do NOT terminate based on the original relative-change test.  The whole
% point of this diagnostic is to see whether a nominally "converged" point
% still improves over a much longer horizon.
force_fixed_iterations = true;

% Save a checkpoint after every diagnosed channel.
save_each_channel = true;

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

param_file = fullfile(project_root, ...
    'DeepSSP_depth_by_G_terminal_gap_g2max3_Fig_data.mat');

base_file = fullfile(fig_dir,'results', ...
    'data_Fig5_DeepSSP_generalization_Pt_Nt_G4_100ch.mat');

deep125_file = fullfile(fig_dir,'results', ...
    'data_Fig5_DeepSSP_generalization_Pt_Nt_G4_N125_supplement_100ch.mat');

required_files = {dataset_file,param_file,base_file,deep125_file};

for k = 1:numel(required_files)
    if ~exist(required_files{k},'file')
        error('Cannot find required file:\n%s',required_files{k});
    end
end

Dset = load(dataset_file);
Plearn = load(param_file);
Base = load(base_file);
Supp = load(deep125_file);

%% -------------------- Configuration -------------------------------------
Pt_dBm_list = double(Base.Pt_dBm_list(:).');
Pt_list = double(Base.Pt_list(:).');

iP30 = find(abs(Pt_dBm_list-target_Pt_dBm)<1e-12,1);

if isempty(iP30)
    error('Pt = %g dBm is not present in Pt_dBm_list.',target_Pt_dBm);
end

iP25 = find(abs(Pt_dBm_list-25)<1e-12,1);

K = double(Base.K);
M = double(Base.M);
N_RF = double(Base.N_RF);
G = double(Base.G);

noise_c = double(Base.noise_c);
noise_s = double(Base.noise_s);
L = double(Base.L);

delta_c = double(Base.delta_c0);
delta_s = double(Base.delta_s0);

Pt30 = double(Pt_list(iP30));

Nth = 8;
Ntv = 8;
Nrh = 10;
Nrv = 8;

Nt = Nth*Ntv;
Z = build_groupconnected_mask(Nt,N_RF,G);

I_out = size(Base.Ptres.ObjRef,2);

%% -------------------- Frozen Deep schedule -------------------------------
Ndeep = 125;

G_learn = double(Plearn.G_list(:).');
idxG = find(G_learn==G,1,'first');

if isempty(idxG)
    error('G=%d not found in trained parameter file.',G);
end

gamma_all = double(Plearn.gamma_B3);
beta_all = double(Plearn.beta_B3);
stage1_layers = double(Plearn.stage1_layers);

if size(gamma_all,2)<Ndeep || size(beta_all,2)<Ndeep
    error('Learned schedule contains fewer than %d layers.',Ndeep);
end

gamma_B3 = gamma_all(idxG,1:Ndeep);
beta_B3  = beta_all(idxG,1:Ndeep);

%% -------------------- Solver settings: identical to Fig. 5 --------------
hbf_outer_iter = 5;
hbf_inner_iter = 2;

max_accept_trials = 20;
rho_in = 2;

b1_accept_tol = 1e-10;
deep_accept_tol = 1e-8;

mu_floor = 1e-8;
mu_margin = 1e-8;

original_tol = 1e-5;

%% -------------------- Rank suspicious channels --------------------------
saved_obj_ref = double(Base.Ptres.ObjRef(iP30,:));
saved_sr_ref = double(Base.Ptres.SRRef(iP30,:));
saved_crlb_ref = double(Base.Ptres.CRLBRef(iP30,:));
saved_ref_iter = double(Base.Ptres.RefIter(iP30,:));

saved_obj_deep = double(Supp.Pt125.Obj(iP30,:));
saved_sr_deep = double(Supp.Pt125.SR(iP30,:));
saved_crlb_deep = double(Supp.Pt125.CRLB(iP30,:));

deep_obj_gain = saved_obj_deep-saved_obj_ref;
deep_sr_gain = saved_sr_deep-saved_sr_ref;

if isempty(manual_channels)

    valid = find(isfinite(deep_obj_gain));

    [~,ord] = sort(deep_obj_gain(valid),'descend');

    n_take = min(n_diag_channels,numel(ord));
    diag_channels = valid(ord(1:n_take));

else

    diag_channels = unique(manual_channels(:).','stable');

    if any(diag_channels<1) || any(diag_channels>I_out)
        error('manual_channels must lie in 1:%d.',I_out);
    end
end

nDiag = numel(diag_channels);

fprintf('============================================================\n');
fprintf('Pt = %g dBm SSP-HBF diagnostic\n',target_Pt_dBm);
fprintf('============================================================\n');
fprintf('G=%d, K=%d, M=%d, Nt=%d, N_RF=%d\n',G,K,M,Nt,N_RF);
fprintf('Weights: delta_c=%.6e, delta_s=%.6e\n',delta_c,delta_s);
fprintf('Extended default SSP iterations = %d\n',ssp30_max_iter);
fprintf('Deep125 -> SSP refinement       = %d\n',deep_refine_iter);
fprintf('Selected channels = [%s]\n\n',num2str(diag_channels));

fprintf('Saved 30-dBm points for selected channels:\n');
fprintf(' ch | refIter |   Obj SSP   | Obj Deep125 | dObj       | SR SSP | SR Deep | dSR\n');

for q = 1:nDiag
    c = diag_channels(q);
    fprintf('%3d | %7.0f | %+10.6e | %+10.6e | %+9.2e | %6.2f | %7.2f | %+6.2f\n', ...
        c,saved_ref_iter(c), ...
        saved_obj_ref(c),saved_obj_deep(c),deep_obj_gain(c), ...
        saved_sr_ref(c),saved_sr_deep(c),deep_sr_gain(c));
end

%% -------------------- Storage -------------------------------------------
Diag = struct();

Diag.channels = diag_channels;
Diag.target_Pt_dBm = target_Pt_dBm;

Diag.saved_obj_ref = saved_obj_ref(diag_channels);
Diag.saved_sr_ref = saved_sr_ref(diag_channels);
Diag.saved_crlb_ref = saved_crlb_ref(diag_channels);
Diag.saved_ref_iter = saved_ref_iter(diag_channels);

Diag.saved_obj_deep125 = saved_obj_deep(diag_channels);
Diag.saved_sr_deep125 = saved_sr_deep(diag_channels);
Diag.saved_crlb_deep125 = saved_crlb_deep(diag_channels);

Diag.default_obj_traj = nan(nDiag,ssp30_max_iter+1);
Diag.default_sr_traj = nan(nDiag,ssp30_max_iter+1);
Diag.default_crlb_traj = nan(nDiag,ssp30_max_iter+1);
Diag.default_accept = nan(nDiag,ssp30_max_iter);

Diag.deep_refine_obj_traj = nan(nDiag,deep_refine_iter+1);
Diag.deep_refine_sr_traj = nan(nDiag,deep_refine_iter+1);
Diag.deep_refine_crlb_traj = nan(nDiag,deep_refine_iter+1);
Diag.deep_refine_accept = nan(nDiag,deep_refine_iter);

Diag.deep_rerun_obj = nan(1,nDiag);
Diag.deep_rerun_sr = nan(1,nDiag);
Diag.deep_rerun_crlb = nan(1,nDiag);
Diag.deep_rerun_reject = nan(1,nDiag);

if do_pt25_continuation
    Diag.pt25_to_30_obj_traj = nan(nDiag,pt25_to_30_refine_iter+1);
    Diag.pt25_to_30_sr_traj = nan(nDiag,pt25_to_30_refine_iter+1);
    Diag.pt25_to_30_crlb_traj = nan(nDiag,pt25_to_30_refine_iter+1);
end

result_dir = fullfile(fig_dir,'results');

if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

save_file = fullfile(result_dir, ...
    'diagnostic_Pt30_SSP_reference.mat');

%% ========================================================================
% Main diagnostic loop
%% ========================================================================
for iq = 1:nDiag

    channel = diag_channels(iq);

    fprintf('\n============================================================\n');
    fprintf('Diagnostic channel %d (%d/%d)\n',channel,iq,nDiag);
    fprintf('============================================================\n');

    H = double(squeeze(Dset.H_test_all(channel,:,1:K)));
    H_seed = H;

    alpha = double(Dset.alpha_test(channel,1:M).');
    theta = double(Dset.theta_test(channel,1:M).');
    phi   = double(Dset.phi_test(channel,1:M).');

    %% Build exact 30-dBm problem and original initialization
    Case30 = build_case( ...
        H,H_seed,alpha,theta,phi, ...
        Nth,Ntv,Nrh,Nrv,Z,Pt30,noise_c,noise_s,L);

    %% ---------------------------------------------------------------
    % Test 1: conventional SSP from ORIGINAL initialization, extended
    %% ---------------------------------------------------------------
    fprintf('1) Running default SSP-HBF at 30 dBm for %d iterations...\n', ...
        ssp30_max_iter);

    Rdef = run_ssp_diagnostic( ...
        Case30,Case30.State0,ssp30_max_iter, ...
        delta_c,delta_s,noise_c,K,M,Pt30, ...
        hbf_outer_iter,hbf_inner_iter, ...
        max_accept_trials,rho_in,b1_accept_tol, ...
        mu_floor,mu_margin,original_tol, ...
        force_fixed_iterations);

    Diag.default_obj_traj(iq,:) = Rdef.Obj;
    Diag.default_sr_traj(iq,:) = Rdef.SR;
    Diag.default_crlb_traj(iq,:) = Rdef.CRLB;
    Diag.default_accept(iq,:) = Rdef.Accepted;

    %% ---------------------------------------------------------------
    % Test 2: reconstruct Deep125 and continue ordinary SSP
    %% ---------------------------------------------------------------
    fprintf('2) Reconstructing Deep-SSP N=125 and refining with SSP...\n');

    Rdeep = run_deep_return_state( ...
        Case30,Case30.State0, ...
        gamma_B3,beta_B3,stage1_layers,Ndeep, ...
        delta_c,delta_s,noise_c,K,M,Pt30, ...
        hbf_outer_iter,hbf_inner_iter, ...
        mu_floor,mu_margin,deep_accept_tol);

    Diag.deep_rerun_obj(iq) = Rdeep.Obj(end);
    Diag.deep_rerun_sr(iq) = Rdeep.SR(end);
    Diag.deep_rerun_crlb(iq) = Rdeep.CRLB(end);
    Diag.deep_rerun_reject(iq) = Rdeep.Reject(end);

    Rrefine = run_ssp_diagnostic( ...
        Case30,Rdeep.StateFinal,deep_refine_iter, ...
        delta_c,delta_s,noise_c,K,M,Pt30, ...
        hbf_outer_iter,hbf_inner_iter, ...
        max_accept_trials,rho_in,b1_accept_tol, ...
        mu_floor,mu_margin,original_tol, ...
        force_fixed_iterations);

    Diag.deep_refine_obj_traj(iq,:) = Rrefine.Obj;
    Diag.deep_refine_sr_traj(iq,:) = Rrefine.SR;
    Diag.deep_refine_crlb_traj(iq,:) = Rrefine.CRLB;
    Diag.deep_refine_accept(iq,:) = Rrefine.Accepted;

    %% ---------------------------------------------------------------
    % Optional Test 3: continuation from 25 dBm
    %% ---------------------------------------------------------------
    if do_pt25_continuation

        if isempty(iP25)
            error('25 dBm is not present in the Pt sweep.');
        end

        Pt25 = double(Pt_list(iP25));

        fprintf('3) Running 25-dBm SSP then continuing at 30 dBm...\n');

        Case25 = build_case( ...
            H,H_seed,alpha,theta,phi, ...
            Nth,Ntv,Nrh,Nrv,Z,Pt25,noise_c,noise_s,L);

        R25 = run_ssp_diagnostic( ...
            Case25,Case25.State0,pt25_ssp_iter, ...
            delta_c,delta_s,noise_c,K,M,Pt25, ...
            hbf_outer_iter,hbf_inner_iter, ...
            max_accept_trials,rho_in,b1_accept_tol, ...
            mu_floor,mu_margin,original_tol, ...
            force_fixed_iterations);

        State25to30 = R25.StateFinal;

        State25to30.Wd = ...
            sqrt(Pt30)/max(norm(State25to30.Wa*State25to30.Wd,'fro'),1e-12) ...
            * State25to30.Wd;

        State25to30.W = State25to30.Wa*State25to30.Wd;

        [State25to30.obj,State25to30.sr,State25to30.crlb,State25to30.FIM] = ...
            evaluate_metrics( ...
                Case30.Pre,H,State25to30.W, ...
                delta_c,delta_s,noise_c,K);

        Rcont = run_ssp_diagnostic( ...
            Case30,State25to30,pt25_to_30_refine_iter, ...
            delta_c,delta_s,noise_c,K,M,Pt30, ...
            hbf_outer_iter,hbf_inner_iter, ...
            max_accept_trials,rho_in,b1_accept_tol, ...
            mu_floor,mu_margin,original_tol, ...
            force_fixed_iterations);

        Diag.pt25_to_30_obj_traj(iq,:) = Rcont.Obj;
        Diag.pt25_to_30_sr_traj(iq,:) = Rcont.SR;
        Diag.pt25_to_30_crlb_traj(iq,:) = Rcont.CRLB;
    end

    %% ---------------------------------------------------------------
    % Channel-wise report
    %% ---------------------------------------------------------------
    idx2000 = min(2000,ssp30_max_iter)+1;

    fprintf('\nChannel %d summary:\n',channel);
    fprintf('  Saved SSP ref      : obj=%+.8e, SR=%.5f, CRLB=%.6e, iter=%g\n', ...
        saved_obj_ref(channel),saved_sr_ref(channel), ...
        saved_crlb_ref(channel),saved_ref_iter(channel));

    fprintf('  Saved Deep125      : obj=%+.8e, SR=%.5f, CRLB=%.6e\n', ...
        saved_obj_deep(channel),saved_sr_deep(channel), ...
        saved_crlb_deep(channel));

    fprintf('  Re-run Deep125     : obj=%+.8e, SR=%.5f, CRLB=%.6e\n', ...
        Rdeep.Obj(end),Rdeep.SR(end),Rdeep.CRLB(end));

    fprintf('  Default SSP @2000 : obj=%+.8e, SR=%.5f, CRLB=%.6e\n', ...
        Rdef.Obj(idx2000),Rdef.SR(idx2000),Rdef.CRLB(idx2000));

    fprintf('  Default SSP @%d : obj=%+.8e, SR=%.5f, CRLB=%.6e\n', ...
        ssp30_max_iter,Rdef.Obj(end),Rdef.SR(end),Rdef.CRLB(end));

    fprintf('  Deep125 + SSP      : obj=%+.8e, SR=%.5f, CRLB=%.6e\n', ...
        Rrefine.Obj(end),Rrefine.SR(end),Rrefine.CRLB(end));

    fprintf('  Gain: extended-default over saved ref = %+.3e\n', ...
        Rdef.Obj(end)-saved_obj_ref(channel));

    fprintf('  Gain: Deep-refined over saved ref     = %+.3e\n', ...
        Rrefine.Obj(end)-saved_obj_ref(channel));

    fprintf('  Deep rerun mismatch in objective      = %+.3e\n', ...
        Rdeep.Obj(end)-saved_obj_deep(channel));

    if save_each_channel
        save(save_file,'Diag', ...
            'ssp30_max_iter','deep_refine_iter', ...
            'do_pt25_continuation','pt25_ssp_iter', ...
            'pt25_to_30_refine_iter', ...
            'delta_c','delta_s','target_Pt_dBm','-v7.3');
    end
end

%% ========================================================================
% Aggregate diagnosis
%% ========================================================================
obj_saved = Diag.saved_obj_ref(:);
obj_deep_saved = Diag.saved_obj_deep125(:);

obj_default_2000 = Diag.default_obj_traj(:,min(2000,ssp30_max_iter)+1);
obj_default_final = Diag.default_obj_traj(:,end);

obj_deep_rerun = Diag.deep_rerun_obj(:);
obj_deep_refined = Diag.deep_refine_obj_traj(:,end);

sr_default_final = Diag.default_sr_traj(:,end);
crlb_default_final = Diag.default_crlb_traj(:,end);

sr_deep_refined = Diag.deep_refine_sr_traj(:,end);
crlb_deep_refined = Diag.deep_refine_crlb_traj(:,end);

gain_after_2000 = obj_default_final-obj_default_2000;
gain_ext_vs_saved = obj_default_final-obj_saved;
gain_deep_refined_vs_saved = obj_deep_refined-obj_saved;

deep_reconstruction_error = obj_deep_rerun-obj_deep_saved;

Diag.gain_after_2000 = gain_after_2000;
Diag.gain_ext_vs_saved = gain_ext_vs_saved;
Diag.gain_deep_refined_vs_saved = gain_deep_refined_vs_saved;
Diag.deep_reconstruction_error = deep_reconstruction_error;

fprintf('\n\n============================================================\n');
fprintf('AGGREGATE DIAGNOSIS OVER %d SELECTED CHANNELS\n',nDiag);
fprintf('============================================================\n');

fprintf('Mean saved SSP objective            = %+.8e\n',mean(obj_saved,'omitnan'));
fprintf('Mean saved Deep125 objective        = %+.8e\n',mean(obj_deep_saved,'omitnan'));
fprintf('Mean default SSP objective @2000    = %+.8e\n',mean(obj_default_2000,'omitnan'));
fprintf('Mean default SSP objective @%d      = %+.8e\n', ...
    ssp30_max_iter,mean(obj_default_final,'omitnan'));
fprintf('Mean Deep125->SSP objective         = %+.8e\n',mean(obj_deep_refined,'omitnan'));

fprintf('\nMean objective gains:\n');
fprintf('  Default SSP: iter 2000 -> %d       = %+.3e\n', ...
    ssp30_max_iter,mean(gain_after_2000,'omitnan'));
fprintf('  Extended default vs saved ref      = %+.3e\n', ...
    mean(gain_ext_vs_saved,'omitnan'));
fprintf('  Deep-refined vs saved ref          = %+.3e\n', ...
    mean(gain_deep_refined_vs_saved,'omitnan'));

fprintf('\nDeep reconstruction consistency:\n');
fprintf('  mean |rerun - saved| objective     = %.3e\n', ...
    mean(abs(deep_reconstruction_error),'omitnan'));
fprintf('  max  |rerun - saved| objective     = %.3e\n', ...
    max(abs(deep_reconstruction_error),[],'omitnan'));

fprintf('\nAverage final physical metrics:\n');
fprintf('  Saved SSP ref    : SR=%.5f, CRLB=%.6e\n', ...
    mean(Diag.saved_sr_ref,'omitnan'), ...
    mean(Diag.saved_crlb_ref,'omitnan'));
fprintf('  Saved Deep125    : SR=%.5f, CRLB=%.6e\n', ...
    mean(Diag.saved_sr_deep125,'omitnan'), ...
    mean(Diag.saved_crlb_deep125,'omitnan'));
fprintf('  Default SSP ext. : SR=%.5f, CRLB=%.6e\n', ...
    mean(sr_default_final,'omitnan'), ...
    mean(crlb_default_final,'omitnan'));
fprintf('  Deep125 -> SSP   : SR=%.5f, CRLB=%.6e\n', ...
    mean(sr_deep_refined,'omitnan'), ...
    mean(crlb_deep_refined,'omitnan'));

%% Automatic qualitative diagnostic
scale_obj = max(1,mean(abs(obj_saved),'omitnan'));
tol_diag = 1e-7*scale_obj;

slow_indicator = mean(gain_after_2000,'omitnan') > tol_diag;
basin_indicator = ...
    mean(obj_deep_refined-obj_default_final,'omitnan') > tol_diag;

fprintf('\n------------------------------------------------------------\n');
fprintf('Automatic diagnostic (use together with the trajectories):\n');

if slow_indicator
    fprintf(['  * The default SSP run still improves materially after ' ...
             'iteration 2000.\n']);
    fprintf('    => The 2000-iteration budget is likely insufficient.\n');
else
    fprintf(['  * The default SSP run changes very little after ' ...
             'iteration 2000.\n']);
end

if basin_indicator
    fprintf(['  * Deep125->SSP converges to a materially better objective ' ...
             'than the extended default SSP run.\n']);
    fprintf('    => A poorer SSP stationary basin is likely involved.\n');
else
    fprintf(['  * Deep125->SSP does not remain materially above the ' ...
             'extended default SSP run.\n']);
end

if max(abs(deep_reconstruction_error),[],'omitnan') > 1e-7
    fprintf(['  * WARNING: reconstructed Deep125 differs noticeably from ' ...
             'the saved result.\n']);
    fprintf('    Check exact data/model consistency before interpreting.\n');
else
    fprintf('  * Deep125 reconstruction agrees with the saved result.\n');
end

fprintf('------------------------------------------------------------\n');

%% ========================================================================
% Figure 1: objective gain over SAVED SSP reference
%% ========================================================================
fig1 = figure('Color','w','Name','Pt=30 SSP diagnostic: objective', ...
    'NumberTitle','off');

hold on;

x_def = 0:ssp30_max_iter;
gain_def = Diag.default_obj_traj-Diag.saved_obj_ref(:);
mean_gain_def = mean(gain_def,1,'omitnan');

x_refine = 0:deep_refine_iter;
gain_refine = Diag.deep_refine_obj_traj-Diag.saved_obj_ref(:);
mean_gain_refine = mean(gain_refine,1,'omitnan');

plot(x_def,mean_gain_def,'LineWidth',1.7);
plot(x_refine,mean_gain_refine,'--','LineWidth',1.7);

xline(2000,':','2000 iterations','LineWidth',1.2);

yline(0,':','Saved SSP reference','LineWidth',1.1);

xlabel('Additional SSP iteration');
ylabel('Objective gain over saved SSP reference');
legend({'Default SSP from original initialization', ...
        'SSP initialized by Deep-SSP N=125'}, ...
       'Location','best','Box','off');

grid on;
box off;
set(gca,'FontName','Times New Roman','FontSize',11);

%% ========================================================================
% Figure 2: average SR-CRLB operating points
%% ========================================================================
fig2 = figure('Color','w','Name','Pt=30 SSP diagnostic: SR-CRLB', ...
    'NumberTitle','off');

hold on;

avg_crlb = [ ...
    mean(Diag.saved_crlb_ref,'omitnan'), ...
    mean(Diag.saved_crlb_deep125,'omitnan'), ...
    mean(crlb_default_final,'omitnan'), ...
    mean(crlb_deep_refined,'omitnan')];

avg_sr = [ ...
    mean(Diag.saved_sr_ref,'omitnan'), ...
    mean(Diag.saved_sr_deep125,'omitnan'), ...
    mean(sr_default_final,'omitnan'), ...
    mean(sr_deep_refined,'omitnan')];

mk = {'s','o','^','d'};
labels = {'Saved SSP ref','Deep-SSP N=125', ...
          sprintf('Default SSP %d',ssp30_max_iter), ...
          sprintf('Deep125 + SSP %d',deep_refine_iter)};

for k = 1:numel(avg_sr)
    plot(avg_crlb(k),avg_sr(k),mk{k}, ...
        'MarkerSize',8,'LineWidth',1.6);
end

xlabel('CRLB');
ylabel('Sum rate (nat/s/Hz)');
legend(labels,'Location','best','Box','off');
grid on;
box off;
set(gca,'FontName','Times New Roman','FontSize',11);

%% Save final diagnostic
save(save_file,'Diag', ...
    'ssp30_max_iter','deep_refine_iter', ...
    'do_pt25_continuation','pt25_ssp_iter', ...
    'pt25_to_30_refine_iter', ...
    'delta_c','delta_s','target_Pt_dBm','-v7.3');

fprintf('\nDiagnostic data saved to:\n%s\n',save_file);
fprintf('============================================================\n');

%% ========================================================================
% Diagnostic-specific local functions
%% ========================================================================
function C = build_case( ...
    H,H_seed,alpha,theta,phi, ...
    Nth,Ntv,Nrh,Nrv,Z,Pt,noise_c,noise_s,L)

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
        error('H_seed must contain at least N_RF columns.');
    end

    Wa0 = Z.*exp(1j*angle(H_seed(:,1:N_RF)));
    Wd0 = digital_update_groupconnected(Wa0,H);
    Wd0 = sqrt(Pt)/max(norm(Wa0*Wd0,'fro'),1e-12)*Wd0;
    W0 = Wa0*Wd0;

    FIM0 = calculateFIM(Pre,W0);

    State0 = struct();
    State0.W = W0;
    State0.Wa = Wa0;
    State0.Wd = Wd0;
    State0.FIM = FIM0;

    C = struct();
    C.Pre = Pre;
    C.H = H;
    C.Nbasis = Nbasis;
    C.R_N = R_N;
    C.Z = Z;
    C.State0 = State0;
end


function R = run_ssp_diagnostic( ...
    C,StateInit,max_iter, ...
    delta_c,delta_s,noise_c,K,M,Pt, ...
    hbf_outer_iter,hbf_inner_iter, ...
    max_accept_trials,rho_in,b1_accept_tol, ...
    mu_floor,mu_margin,tol,force_fixed_iterations)

    Pre = C.Pre;
    H = C.H;
    Nbasis = C.Nbasis;
    R_N = C.R_N;
    Z = C.Z;

    W = StateInit.W;
    Wa = StateInit.Wa;
    Wd = StateInit.Wd;

    [obj_cur,sr_cur,crlb_cur,FIM] = ...
        evaluate_metrics(Pre,H,W,delta_c,delta_s,noise_c,K);

    Obj = nan(1,max_iter+1);
    SR = nan(1,max_iter+1);
    CRLB = nan(1,max_iter+1);
    Accepted = false(1,max_iter);

    Obj(1) = obj_cur;
    SR(1) = sr_cur;
    CRLB(1) = crlb_cur;

    stop_iter = max_iter;

    for t = 1:max_iter

        W_old = W;
        Wa_old = Wa;
        Wd_old = Wd;
        FIM_old = FIM;

        obj_old = obj_cur;
        sr_old = sr_cur;
        crlb_old = crlb_cur;

        [grad_total,lambda_local] = ...
            compute_grad_and_local_curvature( ...
                Pre,H,W_old,FIM_old,Nbasis,R_N, ...
                delta_c,delta_s,noise_c,K,M,mu_floor,mu_margin);

        lambda_trial = lambda_local;
        accepted = false;

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

                break;
            end

            lambda_trial = rho_in*lambda_trial;
        end

        if ~accepted
            W = W_old;
            Wa = Wa_old;
            Wd = Wd_old;
            FIM = FIM_old;

            obj_cur = obj_old;
            sr_cur = sr_old;
            crlb_cur = crlb_old;
        end

        Accepted(t) = accepted;

        Obj(t+1) = obj_cur;
        SR(t+1) = sr_cur;
        CRLB(t+1) = crlb_cur;

        relchg = ...
            abs([obj_cur,crlb_cur,sr_cur]-[obj_old,crlb_old,sr_old]) ./ ...
            max(abs([obj_old,crlb_old,sr_old]),1e-12);

        if ~force_fixed_iterations && all(relchg<=tol)
            stop_iter = t;
            Obj(t+2:end) = obj_cur;
            SR(t+2:end) = sr_cur;
            CRLB(t+2:end) = crlb_cur;
            break;
        end

        if t==1 || mod(t,500)==0 || t==max_iter
            fprintf(['    SSP iter %5d | obj=%+.8e | SR=%8.4f | ' ...
                     'CRLB=%9.3e | accepted=%d\n'], ...
                t,obj_cur,sr_cur,crlb_cur,accepted);
        end
    end

    StateFinal = struct();
    StateFinal.W = W;
    StateFinal.Wa = Wa;
    StateFinal.Wd = Wd;
    StateFinal.FIM = FIM;
    StateFinal.obj = obj_cur;
    StateFinal.sr = sr_cur;
    StateFinal.crlb = crlb_cur;

    R = struct();
    R.Obj = Obj;
    R.SR = SR;
    R.CRLB = CRLB;
    R.Accepted = Accepted;
    R.stop_iter = stop_iter;
    R.StateFinal = StateFinal;
end


function R = run_deep_return_state( ...
    C,StateInit,gamma,beta,stage1_layers,Ndeep, ...
    delta_c,delta_s,noise_c,K,M,Pt, ...
    hbf_outer_iter,hbf_inner_iter, ...
    mu_floor,mu_margin,accept_tol)

    Pre = C.Pre;
    H = C.H;
    Nbasis = C.Nbasis;
    R_N = C.R_N;
    Z = C.Z;

    W = StateInit.W;
    Wa = StateInit.Wa;
    Wd = StateInit.Wd;
    FIM = StateInit.FIM;

    [obj_cur,sr_cur,crlb_cur,~] = ...
        evaluate_metrics(Pre,H,W,delta_c,delta_s,noise_c,K);

    Obj = nan(1,Ndeep+1);
    SR = nan(1,Ndeep+1);
    CRLB = nan(1,Ndeep+1);
    Reject = zeros(1,Ndeep+1);

    Obj(1) = obj_cur;
    SR(1) = sr_cur;
    CRLB(1) = crlb_cur;

    W_prev = W;
    reject_count = 0;

    for t = 1:Ndeep

        W_old = W;
        Wa_old = Wa;
        Wd_old = Wd;

        obj_old = obj_cur;
        sr_old = sr_cur;
        crlb_old = crlb_cur;

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

        inertial_state = W_old+beta_t*(W_old-W_prev);
        effective_shift = lambda_local/max(gamma_t,1e-8);

        W_target = grad_total+effective_shift*inertial_state;

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

        if obj_cand >= obj_old-accept_tol
            W_new = W_cand;
            Wa_new = Wa_cand;
            Wd_new = Wd_cand;
            FIM_new = FIM_cand;

            obj_new = obj_cand;
            sr_new = sr_cand;
            crlb_new = crlb_cand;
        else
            W_new = W_old;
            Wa_new = Wa_old;
            Wd_new = Wd_old;
            FIM_new = FIM;

            obj_new = obj_old;
            sr_new = sr_old;
            crlb_new = crlb_old;

            reject_count = reject_count+1;
        end

        W_prev = W_old;

        W = W_new;
        Wa = Wa_new;
        Wd = Wd_new;
        FIM = FIM_new;

        obj_cur = obj_new;
        sr_cur = sr_new;
        crlb_cur = crlb_new;

        Obj(t+1) = obj_cur;
        SR(t+1) = sr_cur;
        CRLB(t+1) = crlb_cur;
        Reject(t+1) = reject_count;
    end

    StateFinal = struct();
    StateFinal.W = W;
    StateFinal.Wa = Wa;
    StateFinal.Wd = Wd;
    StateFinal.FIM = FIM;
    StateFinal.obj = obj_cur;
    StateFinal.sr = sr_cur;
    StateFinal.crlb = crlb_cur;

    R = struct();
    R.Obj = Obj;
    R.SR = SR;
    R.CRLB = CRLB;
    R.Reject = Reject;
    R.StateFinal = StateFinal;
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
