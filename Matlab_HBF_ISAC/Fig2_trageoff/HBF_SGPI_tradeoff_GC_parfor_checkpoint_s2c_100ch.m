clc
clear
close all

%% ================================================================
% GC-HBF-SGPI tradeoff-region simulation using PARFOR
%
% Parallelization strategy:
%   - Outer architecture loop G: serial
%   - 100 independent channel realizations: PARFOR
%   - Tradeoff-weight sweep inside each channel: serial
%
% Sweep direction:
%   sensing-dominant -> communication-dominant
%
% This preserves the warm-start dependency across neighboring weights.
%
% Architectures:
%   G = 1 : fully connected (FC)
%   G = 2 : group connected
%   G = 4 : group connected
%   G = 8 : sub-connected (SC), for N_RF = 8
%
% This script saves only final tradeoff-region data, not per-iteration
% convergence trajectories. It is intended specifically for the SR-CRLB
% tradeoff figure.
%
% Base MATLAB is used for db-to-power conversion and |.|^2.
% PARFOR requires Parallel Computing Toolbox.
% ================================================================

%% ================================================================
% Load fixed common test dataset
% ================================================================
load fixed_dataset_K8_Nt64_M3.mat

%% ================================================================
% Tradeoff weights
% ================================================================
summary_file = 'tradeoff_summary_K8_N40_outer5_inner2.mat';

if exist(summary_file, 'file')
    Ssum = load(summary_file);

    lambda_list  = double(Ssum.lambda_done(:)).';
    delta_c_list = double(Ssum.delta_c_done(:)).';
    delta_s_list = double(Ssum.delta_s_done(:)).';
else
    warning('Summary file not found. Use default lambda list.');

    lambda_list = unique([ ...
        0, ...
        logspace(-4,2,16), ...
        2e2, 3e2, 5e2, 7e2, ...
        1e3, 2e3, 5e3, 1e4, 1e5, 1e6]);

    delta_c_list = 1 ./ (1 + lambda_list);
    delta_s_list = lambda_list ./ (1 + lambda_list);
end

Nw = length(lambda_list);

% lambda = delta_s/delta_c.
% Descending lambda means sensing-dominant -> communication-dominant.
[~, run_order] = sort(lambda_list, 'descend');

%% ================================================================
% System setup
% ================================================================
M = 3;
K = 8;

% transmit array
Nth = 8;
Ntv = 8;
Nt = Nth * Ntv;

% hybrid architecture
N_RF = K;
G_list = [1, 2, 4, 8];

if any(mod(Nt, G_list) ~= 0) || any(mod(N_RF, G_list) ~= 0)
    error('Every G must divide both Nt and N_RF.');
end

% receive array
Nrh = 10;
Nrv = 8;
Nr = Nrh * Nrv;

% power/noise in linear scale
Pt      = db2power(20);
noise_c = db2power(0);
noise_s = db2power(0);
L = 64;

%% ================================================================
% SSP-HBF algorithm parameters
% ================================================================
tol = 1e-5;
count_max = 2000;

max_accept_trials = 20;
eps_dec = 1e-10;

rho_in = 2;
use_lowdim_exact_mu = true;
mu_floor  = 1e-8;
mu_margin = 1e-8;

%% ================================================================
% Monte Carlo setup
% ================================================================
I_out = 100;

% Restartable checkpoint settings for cluster wall-time limits.
% One checkpoint is written after each PARFOR wave.
resume_from_checkpoint = true;
checkpoint_batch_size = [];  % [] -> use one PARFOR wave = number of workers

if size(H_test_all, 1) < I_out || ...
        size(alpha_test, 1) < I_out || ...
        size(theta_test, 1) < I_out || ...
        size(phi_test, 1) < I_out
    error('The fixed test dataset must contain at least %d realizations.', I_out);
end

save_dir = 'tradeoff_results_GC_parfor';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

%% ================================================================
% Parallel pool
% ================================================================
pool = gcp('nocreate');

if isempty(pool)
    fprintf('Starting parallel pool...\n');
    pool = parpool;
end

fprintf('Parallel pool: %d workers\n', pool.NumWorkers);

if isempty(checkpoint_batch_size)
    checkpoint_batch_size = pool.NumWorkers;
end
checkpoint_batch_size = max(1, round(checkpoint_batch_size));

fprintf('Checkpoint batch size: %d channels\n', checkpoint_batch_size);

%% ================================================================
% Loop over GC architectures
% ================================================================
for iG = 1:length(G_list)

    G = G_list(iG);
    Z = build_groupconnected_mask(Nt, N_RF, G);

    Nant_per_group = Nt / G;
    Nrf_per_group  = N_RF / G;

    assert(all(sum(Z,2) == Nrf_per_group), ...
        'Each antenna must connect to all RF chains in its group.');

    assert(all(sum(Z,1) == Nant_per_group), ...
        'Each RF chain must connect to all antennas in its group.');

    fprintf('\n============================================================\n');
    fprintf('GC-HBF-SGPI tradeoff: G=%d\n', G);
    fprintf('Channels=%d, weights=%d, workers=%d\n', ...
        I_out, Nw, pool.NumWorkers);
    fprintf('Connections: %d RF chains/antenna, %d antennas/RF chain\n', ...
        Nrf_per_group, Nant_per_group);
    fprintf('Warm-start direction: sensing -> communication\n');
    fprintf('============================================================\n');

    %% ------------------------------------------------------------
    % Restartable output arrays and checkpoint file
    % -------------------------------------------------------------
    % Sweep tag prevents this sensing-to-communication run from
    % overwriting or resuming the previous communication-to-sensing run.
    sweep_tag = 's2c';
    sweep_direction = 'sensing_to_communication';

    save_file = fullfile(save_dir, sprintf( ...
        'data_tradeoff_GC_G%d_HBF_SGPI_K%d_NRF%d_100ch_parfor_%s.mat', ...
        G, K, N_RF, sweep_tag));

    SR_all   = nan(I_out, Nw);
    CRB_all  = nan(I_out, Nw);
    Obj_all  = nan(I_out, Nw);
    Time_all = nan(I_out, Nw);
    Iter_all = nan(I_out, Nw);
    AcceptFail_all = false(I_out, Nw);
    completed_channels = false(I_out, 1);

    % Resume safely from a previous checkpoint if it is compatible.
    if resume_from_checkpoint && exist(save_file, 'file')

        fprintf('Checkpoint found for G=%d. Loading...\n', G);
        CP = load(save_file);

        compatible = ...
            isfield(CP, 'G') && double(CP.G) == G && ...
            isfield(CP, 'sweep_tag') && strcmp(CP.sweep_tag, sweep_tag) && ...
            isfield(CP, 'SR_all') && isequal(size(CP.SR_all), [I_out, Nw]) && ...
            isfield(CP, 'CRB_all') && isequal(size(CP.CRB_all), [I_out, Nw]) && ...
            isfield(CP, 'lambda_list') && ...
            length(CP.lambda_list) == Nw && ...
            max(abs(double(CP.lambda_list(:)) - lambda_list(:))) < 1e-12 && ...
            isfield(CP, 'run_order') && ...
            isequal(double(CP.run_order(:)), double(run_order(:)));

        if ~compatible
            error(['Existing checkpoint for G=%d is incompatible with the ' ...
                   'current simulation settings. Rename/delete it or set ' ...
                   'resume_from_checkpoint=false.'], G);
        end

        SR_all   = double(CP.SR_all);
        CRB_all  = double(CP.CRB_all);
        Obj_all  = double(CP.Obj_all);
        Time_all = double(CP.Time_all);
        Iter_all = double(CP.Iter_all);

        if isfield(CP, 'AcceptFail_all')
            AcceptFail_all = logical(CP.AcceptFail_all);
        end

        if isfield(CP, 'completed_channels')
            completed_channels = logical(CP.completed_channels(:));
        else
            % Backward compatibility with an older fully/partly saved file.
            completed_channels = ...
                all(isfinite(SR_all),2) & ...
                all(isfinite(CRB_all),2) & ...
                all(isfinite(Obj_all),2);
        end

        fprintf('Recovered %d/%d completed channels for G=%d.\n', ...
            sum(completed_channels), I_out, G);
    end

    % Static metadata stored in every checkpoint.
    CPout = struct();
    CPout.lambda_list  = lambda_list;
    CPout.delta_c_list = delta_c_list;
    CPout.delta_s_list = delta_s_list;
    CPout.run_order = run_order;
    CPout.sweep_tag = sweep_tag;
    CPout.sweep_direction = sweep_direction;
    CPout.G = G;
    CPout.G_list = G_list;
    CPout.K = K;
    CPout.M = M;
    CPout.Nt = Nt;
    CPout.Nr = Nr;
    CPout.N_RF = N_RF;
    CPout.Z = Z;
    CPout.Nth = Nth;
    CPout.Ntv = Ntv;
    CPout.Nrh = Nrh;
    CPout.Nrv = Nrv;
    CPout.Pt = Pt;
    CPout.noise_c = noise_c;
    CPout.noise_s = noise_s;
    CPout.L = L;
    CPout.I_out = I_out;
    CPout.count_max = count_max;
    CPout.tol = tol;
    CPout.max_accept_trials = max_accept_trials;
    CPout.rho_in = rho_in;
    CPout.checkpoint_batch_size = checkpoint_batch_size;

    pending_channels = find(~completed_channels);

    if isempty(pending_channels)
        fprintf('G=%d is already complete. No channels need to be rerun.\n', G);
    else
        fprintf('G=%d: %d channels remain.\n', G, length(pending_channels));
    end

    %% ============================================================
    % Chunked PARFOR over independent channel realizations
    %
    % If the Roihu job reaches its wall-time during one wave, at most
    % the current wave is lost. The next run resumes from the last
    % successfully written checkpoint.
    % ============================================================
    num_batches = ceil(length(pending_channels) / checkpoint_batch_size);

    for i_batch = 1:num_batches

        i1 = (i_batch-1) * checkpoint_batch_size + 1;
        i2 = min(i_batch * checkpoint_batch_size, length(pending_channels));

        batch_channels = pending_channels(i1:i2);
        nbatch = length(batch_channels);

        fprintf('\nG=%d | batch %d/%d | channels %d ... %d\n', ...
            G, i_batch, num_batches, batch_channels(1), batch_channels(end));

        % Batch-local sliced arrays. Nothing is written to disk by workers.
        SR_batch   = nan(nbatch, Nw);
        CRB_batch  = nan(nbatch, Nw);
        Obj_batch  = nan(nbatch, Nw);
        Time_batch = nan(nbatch, Nw);
        Iter_batch = nan(nbatch, Nw);
        Fail_batch = false(nbatch, Nw);

        parfor ib = 1:nbatch

            channel = batch_channels(ib);

        % Local output arrays for this channel.
        % Assign once to sliced arrays after the complete weight sweep.
        SR_ch   = nan(1, Nw);
        CRB_ch  = nan(1, Nw);
        Obj_ch  = nan(1, Nw);
        Time_ch = nan(1, Nw);
        Iter_ch = nan(1, Nw);

        Fail_ch = false(1, Nw);

        %% --------------------------------------------------------
        % Channel realization
        % ---------------------------------------------------------
        H = double(squeeze(H_test_all(channel, :, 1:K)));

        alpha = double(alpha_test(channel, 1:M).');
        theta = double(theta_test(channel, 1:M).');
        phi   = double(phi_test(channel, 1:M).');

        %% steering matrices
        [A, dAtheta, dAphi] = ...
            construct_steer_matrix_and_derivative_steer_matrix( ...
            theta, phi, Nth, Ntv);

        [B, dBtheta, dBphi] = ...
            construct_steer_matrix_and_derivative_steer_matrix( ...
            theta, phi, Nrh, Nrv);

        Gamma = diag(alpha);

        %% sensing precomputation
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

        %% compact-gradient basis
        Nbasis = [H, A, dAtheta, dAphi];
        [~, R_N, ~] = qr_rank_basis(Nbasis);

        %% --------------------------------------------------------
        % Warm-start state, private to this channel
        % ---------------------------------------------------------
        Wa_ws = [];
        Wd_ws = [];

        %% ========================================================
        % SERIAL sweep over tradeoff weights
        % =========================================================
        for idx_run = 1:Nw

            iw = run_order(idx_run);

            delta_c = delta_c_list(iw);
            delta_s = delta_s_list(iw);

            t_weight = tic;

            %% ----------------------------------------------------
            % Initialization / warm start
            % -----------------------------------------------------
            if isempty(Wa_ws)

                % First weight: deterministic GC-feasible MRT-like init.
                Wa = Z .* exp(1j * angle(H(:,1:N_RF)));

                Wd = digital_update_groupconnected( ...
                    Wa, H(:,1:K));

                Wd = sqrt(Pt) / ...
                    max(norm(Wa * Wd, 'fro'), 1e-12) * Wd;

                W = Wa * Wd;

            else

                % Neighboring tradeoff point: use previous optimum.
                Wa = Z .* exp(1j * angle(Wa_ws));
                Wd = Wd_ws;

                W = Wa * Wd;

                Wd = sqrt(Pt) / ...
                    max(norm(W, 'fro'), 1e-12) * Wd;

                W = Wa * Wd;
            end

            %% initial objective
            FIM = calculateFIM(Pre, W);

            T_k = sum(square_abs(H' * W), 2) ...
                + noise_c * ones(K,1);

            rate_p = log(T_k ./ ...
                (T_k - square_abs(diag(H' * W))));

            obj_sen = trace(inv(FIM));
            obj_com = sum(rate_p);
            obj_tot = delta_c * obj_com - delta_s * obj_sen;

            Con_prev = [obj_tot, obj_sen, obj_com];

            last_iter = 1;
            accept_failed = false;

            %% ====================================================
            % SSP-HBF optimization
            % =====================================================
            for iter = 2:count_max

                W_prev   = W;
                Wa_prev  = Wa;
                Wd_prev  = Wd;
                FIM_prev = FIM;

                obj_prev = Con_prev(1);

                %% communication part
                T_k = sum(square_abs(H' * W_prev), 2) ...
                    + noise_c * ones(K,1);

                alpha_k = T_k ./ ...
                    (T_k - square_abs(diag(H' * W_prev))) - 1;

                beta_k = sqrt(1 + alpha_k) .* ...
                    diag(H' * W_prev) ./ T_k;

                Sigma1 = diag(sqrt(1 + alpha_k) .* beta_k);
                Sigma2 = diag(square_abs(beta_k));

                %% sensing part
                CRBM = inv(FIM_prev);
                Phi  = CRBM * CRBM;

                Sigma3 = construct_Sigma3(Pre, Phi);

                Omega1 = [ ...
                    delta_c * Sigma1; ...
                    zeros(3*M, K)];

                Omega2 = blkdiag( ...
                    -delta_c * Sigma2, ...
                     delta_s * Sigma3);

                %% reduced exact SSP shift
                if use_lowdim_exact_mu
                    [mu_exact, ~] = ...
                        compute_lowdim_exact_shift_from_Omega2( ...
                        R_N, Omega2);

                    mu = max(mu_exact + mu_margin, mu_floor);
                else
                    mu = 1e-3;
                end

                %% compact SSP target
                grad_total = Nbasis * ...
                    (Omega1 + Omega2 * (Nbasis' * W_prev));

                %% accept/reject loop
                accepted = false;
                trial = 0;

                while ~accepted && trial < max_accept_trials

                    trial = trial + 1;

                    W_target = grad_total + mu * W_prev;

                    [~, Wa_cand, Wd_cand] = ...
                        matrix_approximation_groupconnected( ...
                        W_target, Wa_prev, Wd_prev, Z);

                    % full-power normalization
                    Wd_cand = sqrt(Pt) / ...
                        max(norm(Wa_cand * Wd_cand, 'fro'), 1e-12) ...
                        * Wd_cand;

                    W_cand = Wa_cand * Wd_cand;

                    %% candidate objective
                    FIM_cand = calculateFIM(Pre, W_cand);

                    T_k_cand = ...
                        sum(square_abs(H' * W_cand), 2) ...
                        + noise_c * ones(K,1);

                    rate_p_cand = log(T_k_cand ./ ...
                        (T_k_cand - ...
                        square_abs(diag(H' * W_cand))));

                    obj_sen_cand = trace(inv(FIM_cand));
                    obj_com_cand = sum(rate_p_cand);

                    obj_tot_cand = ...
                        delta_c * obj_com_cand ...
                        - delta_s * obj_sen_cand;

                    if obj_tot_cand >= obj_prev - eps_dec

                        accepted = true;

                        W   = W_cand;
                        Wa  = Wa_cand;
                        Wd  = Wd_cand;
                        FIM = FIM_cand;

                        Con_new = [ ...
                            obj_tot_cand, ...
                            obj_sen_cand, ...
                            obj_com_cand];

                        last_iter = iter;

                    else
                        mu = rho_in * mu;
                    end
                end

                if ~accepted
                    accept_failed = true;
                    break;
                end

                %% stopping test
                relchg = abs(Con_new - Con_prev) ./ ...
                    max(abs(Con_prev), 1e-12);

                Con_prev = Con_new;

                if all(relchg <= tol)
                    break;
                end
            end

            %% ----------------------------------------------------
            % Final performance at this tradeoff point
            % -----------------------------------------------------
            FIM = calculateFIM(Pre, W);
            CRB = trace(inv(FIM));

            T_k = sum(square_abs(H' * W), 2) ...
                + noise_c * ones(K,1);

            SR = sum(log(T_k ./ ...
                (T_k - square_abs(diag(H' * W)))));

            OBJ = delta_c * SR - delta_s * CRB;

            SR_ch(iw)   = SR;
            CRB_ch(iw)  = CRB;
            Obj_ch(iw)  = OBJ;
            Time_ch(iw) = toc(t_weight);
            Iter_ch(iw) = last_iter;

            Fail_ch(iw) = accept_failed;

            %% warm start for the NEXT neighboring weight
            Wa_ws = Wa;
            Wd_ws = Wd;

        end

        %% --------------------------------------------------------
        % PARFOR sliced assignments: once per channel
        % ---------------------------------------------------------
        SR_batch(ib,:)   = SR_ch;
        CRB_batch(ib,:)  = CRB_ch;
        Obj_batch(ib,:)  = Obj_ch;
        Time_batch(ib,:) = Time_ch;
        Iter_batch(ib,:) = Iter_ch;

        Fail_batch(ib,:) = Fail_ch;

        fprintf('G=%d: channel %3d/%3d finished.\n', ...
            G, channel, I_out);

        end

        %% --------------------------------------------------------
        % Commit completed wave to the client-side global arrays
        % ---------------------------------------------------------
        for ib = 1:nbatch
            channel = batch_channels(ib);

            SR_all(channel,:)   = SR_batch(ib,:);
            CRB_all(channel,:)  = CRB_batch(ib,:);
            Obj_all(channel,:)  = Obj_batch(ib,:);
            Time_all(channel,:) = Time_batch(ib,:);
            Iter_all(channel,:) = Iter_batch(ib,:);
            AcceptFail_all(channel,:) = Fail_batch(ib,:);

            completed_channels(channel) = true;
        end

        %% --------------------------------------------------------
        % Checkpoint after EVERY completed PARFOR wave
        % ---------------------------------------------------------
        SR_avg   = mean_no_nan_dim1(SR_all);
        CRB_avg  = mean_no_nan_dim1(CRB_all);
        OBJ_avg  = mean_no_nan_dim1(Obj_all);
        Time_avg = mean_no_nan_dim1(Time_all);
        Iter_avg = mean_no_nan_dim1(Iter_all);

        CPout.SR_all   = SR_all;
        CPout.CRB_all  = CRB_all;
        CPout.Obj_all  = Obj_all;
        CPout.Time_all = Time_all;
        CPout.Iter_all = Iter_all;
        CPout.AcceptFail_all = AcceptFail_all;

        CPout.SR_avg   = SR_avg;
        CPout.CRB_avg  = CRB_avg;
        CPout.OBJ_avg  = OBJ_avg;
        CPout.Time_avg = Time_avg;
        CPout.Iter_avg = Iter_avg;

        CPout.completed_channels = completed_channels;
        CPout.num_completed = sum(completed_channels);
        CPout.is_complete = all(completed_channels);
        CPout.last_checkpoint = datestr(now, 31);

        atomic_save_struct(save_file, CPout);

        fprintf(['CHECKPOINT SAVED: G=%d, completed %d/%d channels ' ...
                 '(%s)\n'], ...
            G, sum(completed_channels), I_out, CPout.last_checkpoint);
    end

    %% ============================================================
    % Finalize this architecture
    % ==============================================================
    SR_avg   = mean_no_nan_dim1(SR_all);
    CRB_avg  = mean_no_nan_dim1(CRB_all);
    OBJ_avg  = mean_no_nan_dim1(Obj_all);
    Time_avg = mean_no_nan_dim1(Time_all);
    Iter_avg = mean_no_nan_dim1(Iter_all);

    CPout.SR_all   = SR_all;
    CPout.CRB_all  = CRB_all;
    CPout.Obj_all  = Obj_all;
    CPout.Time_all = Time_all;
    CPout.Iter_all = Iter_all;
    CPout.AcceptFail_all = AcceptFail_all;

    CPout.SR_avg   = SR_avg;
    CPout.CRB_avg  = CRB_avg;
    CPout.OBJ_avg  = OBJ_avg;
    CPout.Time_avg = Time_avg;
    CPout.Iter_avg = Iter_avg;

    CPout.completed_channels = completed_channels;
    CPout.num_completed = sum(completed_channels);
    CPout.is_complete = all(completed_channels);
    CPout.last_checkpoint = datestr(now,31);

    atomic_save_struct(save_file, CPout);

    fprintf('\nSaved G=%d results/checkpoint to:\n%s\n', G, save_file);
    fprintf('Completed channels: %d/%d\n', ...
        sum(completed_channels), I_out);

    if all(completed_channels)
        fprintf('G=%d COMPLETE.\n', G);
        fprintf('Mean failures per weight: %.3f / 100 channels\n', ...
            mean(sum(AcceptFail_all,1)));
    else
        fprintf('G=%d remains PARTIAL and will resume next run.\n', G);
    end

end

fprintf('\n============================================================\n');
fprintf('All GC tradeoff-region simulations completed.\n');
fprintf('============================================================\n');



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


function m = mean_no_nan_dim1(X)
%MEAN_NO_NAN_DIM1 Mean over dimension 1 while ignoring NaNs.
% Base-MATLAB implementation for compatibility.

    valid = isfinite(X);
    X0 = X;
    X0(~valid) = 0;

    count = sum(valid,1);
    total = sum(X0,1);

    m = total ./ max(count,1);
    m(count == 0) = NaN;
end


function atomic_save_struct(save_file, S)
%ATOMIC_SAVE_STRUCT Robust checkpoint write.
%
% Save to a temporary MAT file first. Only after the save succeeds is the
% old checkpoint replaced. Therefore, if the cluster job is killed during
% SAVE, the previous valid checkpoint remains intact.

    tmp_file = [save_file '.tmp.mat'];

    % Remove a stale temporary file left by a previously interrupted job.
    if exist(tmp_file, 'file')
        delete(tmp_file);
    end

    save(tmp_file, '-struct', 'S', '-v7.3');

    [ok, msg] = movefile(tmp_file, save_file, 'f');

    if ~ok
        error('Could not commit checkpoint: %s', msg);
    end
end

