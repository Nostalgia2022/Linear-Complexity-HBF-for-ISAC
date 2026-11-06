clc
clear
close all

%% ================================================================
% Fully-digital SGPI tradeoff simulation using PARFOR + checkpoints
%
% Parallelization:
%   channel realizations -> PARFOR
%   tradeoff weights     -> serial warm-start continuation
%
% Sweep direction:
%   communication-dominant -> sensing-dominant (C2S)
%
% Rerunning this file resumes from the last completed channel batch.
% ================================================================

load fixed_dataset_K8_Nt64_M3.mat

%% System setup
M = 3;
K = 8;

Nth = 8;
Ntv = 8;
Nt = Nth*Ntv;

Nrh = 10;
Nrv = 8;
Nr = Nrh*Nrv;

Pt      = db2power(20);
noise_c = db2power(0);
noise_s = db2power(0);
L = 64;

%% Algorithm parameters
tolerance = 1e-5;
count_max = 2000;
I_out = 100;

if size(H_test_all,1) < I_out || size(alpha_test,1) < I_out || ...
        size(theta_test,1) < I_out || size(phi_test,1) < I_out
    error('The fixed dataset must contain at least %d test channels.',I_out);
end

%% Tradeoff weights -- same grid as HBF simulations
summary_file = 'tradeoff_summary_K8_N40_outer5_inner2.mat';

if exist(summary_file,'file')
    Ssum = load(summary_file);
    lambda_list  = double(Ssum.lambda_done(:)).';
    delta_c_list = double(Ssum.delta_c_done(:)).';
    delta_s_list = double(Ssum.delta_s_done(:)).';
else
    warning('Summary file not found. Use default HBF tradeoff grid.');

    lambda_list = unique([ ...
        0, ...
        logspace(-4,2,16), ...
        2e2, 3e2, 5e2, 7e2, ...
        1e3, 2e3, 5e3, 1e4, 1e5, 1e6]);

    delta_c_list = 1 ./ (1 + lambda_list);
    delta_s_list = lambda_list ./ (1 + lambda_list);
end

Nw = length(lambda_list);

% lambda = delta_s/delta_c:
% ascending lambda = communication -> sensing
[~,run_order] = sort(lambda_list,'ascend');

%% Save/checkpoint settings
save_dir = 'tradeoff_results_FD_parfor';
if ~exist(save_dir,'dir')
    mkdir(save_dir);
end

sweep_tag = 'c2s';
sweep_direction = 'communication_to_sensing';

save_file = fullfile(save_dir,sprintf( ...
    'data_tradeoff_FD_SGPI_K%d_Nt%d_100ch_parfor_%s.mat', ...
    K,Nt,sweep_tag));

resume_from_checkpoint = true;
checkpoint_batch_size = [];   % [] -> number of active workers

%% Parallel/serial execution mode
%
% Try to reuse/start a parallel pool. If the shared PCT license is not
% available on Roihu, continue automatically in serial mode using the
% SAME checkpoint file. A later rerun can resume the unfinished channels
% in parallel when a license becomes available.
use_parallel = false;
pool = [];

try
    pool = gcp('nocreate');

    if isempty(pool)
        fprintf('Trying to start parallel pool...\n');
        pool = parpool;
    end

    use_parallel = true;
    fprintf('Parallel mode enabled: %d workers\n',pool.NumWorkers);

catch ME
    fprintf('\n============================================================\n');
    warning(['Parallel pool unavailable. Falling back to SERIAL mode.\n' ...
             'Reason: %s'], ME.message);
    fprintf('The same checkpoint file will be used.\n');
    fprintf('You can rerun later and resume in parallel.\n');
    fprintf('============================================================\n\n');

    use_parallel = false;
    pool = [];
end

if isempty(checkpoint_batch_size)
    if use_parallel
        checkpoint_batch_size = pool.NumWorkers;
    else
        % In serial mode checkpoint after every completed channel.
        checkpoint_batch_size = 1;
    end
end

checkpoint_batch_size = max(1,round(checkpoint_batch_size));

if use_parallel
    fprintf('Checkpoint batch size: %d channels\n',checkpoint_batch_size);
else
    fprintf('Serial fallback: checkpoint every %d channel(s).\n', ...
        checkpoint_batch_size);
end

%% Restartable storage
SR_all   = nan(I_out,Nw);
CRB_all  = nan(I_out,Nw);
Obj_all  = nan(I_out,Nw);
Time_all = nan(I_out,Nw);
Iter_all = nan(I_out,Nw);

completed_channels = false(I_out,1);

%% Resume compatible checkpoint
if resume_from_checkpoint && exist(save_file,'file')

    fprintf('\nCheckpoint found:\n%s\n',save_file);
    CP = load(save_file);

    compatible = ...
        isfield(CP,'sweep_tag') && strcmp(CP.sweep_tag,sweep_tag) && ...
        isfield(CP,'SR_all') && isequal(size(CP.SR_all),[I_out,Nw]) && ...
        isfield(CP,'CRB_all') && isequal(size(CP.CRB_all),[I_out,Nw]) && ...
        isfield(CP,'lambda_list') && length(CP.lambda_list)==Nw && ...
        max(abs(double(CP.lambda_list(:))-lambda_list(:))) < 1e-12 && ...
        isfield(CP,'run_order') && ...
        isequal(double(CP.run_order(:)),double(run_order(:)));

    if ~compatible
        error(['Existing FD checkpoint is incompatible with current settings. ' ...
               'Rename/delete it or set resume_from_checkpoint=false.']);
    end

    SR_all   = double(CP.SR_all);
    CRB_all  = double(CP.CRB_all);
    Obj_all  = double(CP.Obj_all);
    Time_all = double(CP.Time_all);

    if isfield(CP,'Iter_all')
        Iter_all = double(CP.Iter_all);
    end

    if isfield(CP,'completed_channels')
        completed_channels = logical(CP.completed_channels(:));
    else
        completed_channels = ...
            all(isfinite(SR_all),2) & ...
            all(isfinite(CRB_all),2) & ...
            all(isfinite(Obj_all),2);
    end

    fprintf('Recovered %d/%d completed channels.\n', ...
        sum(completed_channels),I_out);
end

%% Static checkpoint metadata
CPout = struct();
CPout.lambda_list = lambda_list;
CPout.delta_c_list = delta_c_list;
CPout.delta_s_list = delta_s_list;
CPout.run_order = run_order;
CPout.sweep_tag = sweep_tag;
CPout.sweep_direction = sweep_direction;

CPout.M = M;
CPout.K = K;
CPout.Nt = Nt;
CPout.Nr = Nr;
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
CPout.tolerance = tolerance;
CPout.checkpoint_batch_size = checkpoint_batch_size;
CPout.use_parallel_last_run = use_parallel;

%% Pending channels
pending_channels = find(~completed_channels);

if isempty(pending_channels)
    fprintf('\nAll channels are already complete.\n');
else
    fprintf('\n%d/%d channels remain.\n',length(pending_channels),I_out);
end

num_batches = ceil(length(pending_channels)/checkpoint_batch_size);

%% Chunked parallel/serial execution
for i_batch = 1:num_batches

    i1 = (i_batch-1)*checkpoint_batch_size + 1;
    i2 = min(i_batch*checkpoint_batch_size,length(pending_channels));

    batch_channels = pending_channels(i1:i2);
    nbatch = length(batch_channels);

    fprintf('\nFD batch %d/%d | channels %d ... %d\n', ...
        i_batch,num_batches,batch_channels(1),batch_channels(end));

    SR_batch   = nan(nbatch,Nw);
    CRB_batch  = nan(nbatch,Nw);
    Obj_batch  = nan(nbatch,Nw);
    Time_batch = nan(nbatch,Nw);
    Iter_batch = nan(nbatch,Nw);

    if use_parallel

        parfor ib = 1:nbatch

            channel = batch_channels(ib);

            [SR_batch(ib,:),CRB_batch(ib,:),Obj_batch(ib,:), ...
                Time_batch(ib,:),Iter_batch(ib,:)] = ...
                run_fd_channel_tradeoff( ...
                    channel, ...
                    H_test_all,alpha_test,theta_test,phi_test, ...
                    M,K,Nth,Ntv,Nrh,Nrv,Nt, ...
                    L,noise_s,noise_c,Pt, ...
                    Nw,run_order,delta_c_list,delta_s_list, ...
                    count_max,tolerance);
        end

    else

        for ib = 1:nbatch

            channel = batch_channels(ib);

            fprintf('Serial channel %d/%d\n',channel,I_out);

            [SR_batch(ib,:),CRB_batch(ib,:),Obj_batch(ib,:), ...
                Time_batch(ib,:),Iter_batch(ib,:)] = ...
                run_fd_channel_tradeoff( ...
                    channel, ...
                    H_test_all,alpha_test,theta_test,phi_test, ...
                    M,K,Nth,Ntv,Nrh,Nrv,Nt, ...
                    L,noise_s,noise_c,Pt, ...
                    Nw,run_order,delta_c_list,delta_s_list, ...
                    count_max,tolerance);
        end
    end

    %% Commit completed batch
    for ib = 1:nbatch
        channel = batch_channels(ib);

        SR_all(channel,:)   = SR_batch(ib,:);
        CRB_all(channel,:)  = CRB_batch(ib,:);
        Obj_all(channel,:)  = Obj_batch(ib,:);
        Time_all(channel,:) = Time_batch(ib,:);
        Iter_all(channel,:) = Iter_batch(ib,:);

        completed_channels(channel) = true;
    end

    %% Current averages
    SR_avg   = mean_no_nan_dim1(SR_all);
    CRB_avg  = mean_no_nan_dim1(CRB_all);
    Obj_avg  = mean_no_nan_dim1(Obj_all);
    Time_avg = mean_no_nan_dim1(Time_all);
    Iter_avg = mean_no_nan_dim1(Iter_all);

    %% Checkpoint
    CPout.SR_all = SR_all;
    CPout.CRB_all = CRB_all;
    CPout.Obj_all = Obj_all;
    CPout.Time_all = Time_all;
    CPout.Iter_all = Iter_all;

    CPout.SR_avg = SR_avg;
    CPout.CRB_avg = CRB_avg;
    CPout.Obj_avg = Obj_avg;
    CPout.Time_avg = Time_avg;
    CPout.Iter_avg = Iter_avg;

    CPout.completed_channels = completed_channels;
    CPout.num_completed = sum(completed_channels);
    CPout.is_complete = all(completed_channels);
    CPout.last_checkpoint = datestr(now,31);

    atomic_save_struct(save_file,CPout);

    fprintf('CHECKPOINT SAVED: %d/%d channels (%s)\n', ...
        CPout.num_completed,I_out,CPout.last_checkpoint);
end

%% Finalize
SR_avg   = mean_no_nan_dim1(SR_all);
CRB_avg  = mean_no_nan_dim1(CRB_all);
Obj_avg  = mean_no_nan_dim1(Obj_all);
Time_avg = mean_no_nan_dim1(Time_all);
Iter_avg = mean_no_nan_dim1(Iter_all);

CPout.SR_all = SR_all;
CPout.CRB_all = CRB_all;
CPout.Obj_all = Obj_all;
CPout.Time_all = Time_all;
CPout.Iter_all = Iter_all;

CPout.SR_avg = SR_avg;
CPout.CRB_avg = CRB_avg;
CPout.Obj_avg = Obj_avg;
CPout.Time_avg = Time_avg;
CPout.Iter_avg = Iter_avg;

CPout.completed_channels = completed_channels;
CPout.num_completed = sum(completed_channels);
CPout.is_complete = all(completed_channels);
CPout.last_checkpoint = datestr(now,31);

atomic_save_struct(save_file,CPout);

fprintf('\n============================================================\n');
fprintf('Fully-digital SGPI C2S simulation status\n');
fprintf('Completed channels: %d/%d\n',sum(completed_channels),I_out);
fprintf('Saved to:\n%s\n',save_file);

if all(completed_channels)
    fprintf('STATUS: COMPLETE\n');
else
    fprintf('STATUS: PARTIAL -- rerun this same script to resume.\n');
end
fprintf('============================================================\n');

function [SR_ch,CRB_ch,Obj_ch,Time_ch,Iter_ch] = ...
    run_fd_channel_tradeoff( ...
    channel,H_test_all,alpha_test,theta_test,phi_test, ...
    M,K,Nth,Ntv,Nrh,Nrv,Nt, ...
    L,noise_s,noise_c,Pt, ...
    Nw,run_order,delta_c_list,delta_s_list, ...
    count_max,tolerance)
%RUN_FD_CHANNEL_TRADEOFF Fully-digital C2S sweep for one channel.
%
% This function is used by BOTH PARFOR and serial fallback, ensuring that
% the numerical algorithm is identical in the two execution modes.

    
    
    SR_ch   = nan(1,Nw);
    CRB_ch  = nan(1,Nw);
    Obj_ch  = nan(1,Nw);
    Time_ch = nan(1,Nw);
    Iter_ch = nan(1,Nw);
    
    %% Channel realization
    H = double(squeeze(H_test_all(channel,:,1:K)));
    alpha = double(alpha_test(channel,1:M).');
    theta = double(theta_test(channel,1:M).');
    phi   = double(phi_test(channel,1:M).');
    
    [A,dAtheta,dAphi] = ...
        construct_steer_matrix_and_derivative_steer_matrix( ...
        theta,phi,Nth,Ntv);
    
    [B,dBtheta,dBphi] = ...
        construct_steer_matrix_and_derivative_steer_matrix( ...
        theta,phi,Nrh,Nrv);
    
    U = diag(alpha);
    
    %% Warm-start state
    W_ws = [];
    
    %% Serial weight sweep
    for idx_run = 1:Nw
    
        iw = run_order(idx_run);
    
        delta_c = delta_c_list(iw);
        delta_s = delta_s_list(iw);
    
        t_weight = tic;
    
        %% Initialization / warm start
        if isempty(W_ws)
            W = H;
            W = W * sqrt(Pt / max(real(trace(W*W')),1e-12));
        else
            W = W_ws;
            W = W * sqrt(Pt / max(real(trace(W*W')),1e-12));
        end
    
        %% Initial objective
        FIM = calculateFIM( ...
            L,noise_s,W,A,dAtheta,dAphi,B,dBtheta,dBphi,U);
    
        T_k = sum(square_abs(H'*W),2) + noise_c*ones(K,1);
        rate_p = log(T_k ./ (T_k-square_abs(diag(H'*W))));
    
        obj_sen = trace(inv(FIM));
        obj_com = sum(rate_p);
        obj_tot = delta_c*obj_com-delta_s*obj_sen;
    
        Con_prev = [obj_tot,obj_sen,obj_com];
        last_iter = 0;
    
        %% Fully-digital SGPI
        for count = 1:count_max
    
            % Communication part
            T_k = sum(square_abs(H'*W),2) + noise_c*ones(K,1);
    
            alpha_k = T_k ./ ...
                (T_k-square_abs(diag(H'*W))) - 1;
    
            beta_k = sqrt(1+alpha_k) .* diag(H'*W) ./ T_k;
    
            Sigma1 = diag(sqrt(1+alpha_k).*beta_k);
            Sigma2 = diag(square_abs(beta_k));
    
            % Sensing part
            CRBM = inv(FIM);
    
            Q = construct_matrixQ( ...
                L,noise_s,CRBM*CRBM, ...
                A,dAtheta,dAphi,B,dBtheta,dBphi,U);
    
            % SGPI coefficient matrices
            C1 = delta_c * H * Sigma1;
    
            C2 = 0.5*delta_s*(Q+Q') ...
                - delta_c*H*Sigma2*H';
    
            % Numerical Hermitian cleanup
            C2 = (C2+C2')/2;
    
            % Exact dominant-magnitude eigenvalue shift.
            % Nt=64, so full eig is cheap and robust.
            eig_C2 = real(eig(C2));
            mu = max(abs(eig_C2));
    
            % SGPI update
            W = C1 + (mu*eye(Nt)+C2)*W;
    
            W = W * sqrt(Pt / max(real(trace(W*W')),1e-12));
    
            % Evaluate objective
            FIM = calculateFIM( ...
                L,noise_s,W,A,dAtheta,dAphi,B,dBtheta,dBphi,U);
    
            T_k = sum(square_abs(H'*W),2) + noise_c*ones(K,1);
            rate_p = log(T_k ./ ...
                (T_k-square_abs(diag(H'*W))));
    
            obj_sen = trace(inv(FIM));
            obj_com = sum(rate_p);
            obj_tot = delta_c*obj_com-delta_s*obj_sen;
    
            Con_new = [obj_tot,obj_sen,obj_com];
            last_iter = count;
    
            relchg = abs(Con_new-Con_prev) ./ ...
                max(abs(Con_prev),1e-12);
    
            Con_prev = Con_new;
    
            if all(relchg <= tolerance)
                break;
            end
        end
    
        %% Final metrics
        FIM = calculateFIM( ...
            L,noise_s,W,A,dAtheta,dAphi,B,dBtheta,dBphi,U);
    
        CRB = trace(inv(FIM));
    
        T_k = sum(square_abs(H'*W),2) + noise_c*ones(K,1);
        SR = sum(log(T_k ./ ...
            (T_k-square_abs(diag(H'*W)))));
    
        OBJ = delta_c*SR-delta_s*CRB;
    
        SR_ch(iw)   = SR;
        CRB_ch(iw)  = CRB;
        Obj_ch(iw)  = OBJ;
        Time_ch(iw) = toc(t_weight);
        Iter_ch(iw) = last_iter;
    
        W_ws = W;
    end

end


%% ================================================================
% Toolbox-free helpers
% ================================================================

function y = db2power(x)
    y = 10.^(x/10);
end

function y = square_abs(x)
    y = abs(x).^2;
end

function m = mean_no_nan_dim1(X)

    valid = isfinite(X);
    X0 = X;
    X0(~valid) = 0;

    count = sum(valid,1);
    total = sum(X0,1);

    m = total ./ max(count,1);
    m(count==0) = NaN;
end

function atomic_save_struct(save_file,S)

    tmp_file = [save_file '.tmp.mat'];

    if exist(tmp_file,'file')
        delete(tmp_file);
    end

    save(tmp_file,'-struct','S','-v7.3');

    [ok,msg] = movefile(tmp_file,save_file,'f');

    if ~ok
        error('Could not commit checkpoint: %s',msg);
    end
end


function Q=construct_matrixQ(L,noise_s,Phi,A,dAtheta,dAphi,B,dBtheta,dBphi,U)
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


function FIM=calculateFIM(L,noise_s,W,A,dAtheta,dAphi,B,dBtheta,dBphi,U)

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
