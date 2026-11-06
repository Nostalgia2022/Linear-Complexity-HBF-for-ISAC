clc
clear
close all

%% ========================================================================
% System-level SR-CRLB tradeoff
%
% Curves:
%   Digital benchmark
%   SSP-HBF      : G = 1, 2, 4, 8
%   Deep-SSP-HBF : G = 1, 2, 4, 8, fixed practical depth N = 125
%
% Visual rule for this figure:
%   * Connectivity G is encoded by COLOR + MARKER.
%   * SSP-HBF uses the standard/darker color with SOLID line.
%   * Deep-SSP-HBF uses the LIGHTWEIGHT counterpart color and DASHED line.
%     Example: color(1) -> SSP, color(11) -> Deep counterpart.
%   * SSP and Deep for the same G use the SAME marker.
%   * Digital is an independent black benchmark.
%
% This is an intentional exception to the global "solid-line default":
% the dashed line has semantic meaning here -- low-complexity counterpart.
%
% Requires:
%   plot_style.m
%   user's color.m function, if available.
%
% If color.m is unavailable, this script uses fallback colors and creates
% lighter counterparts automatically.
%% ========================================================================

%% -------------------- User options --------------------
G_list = [1 2 4 8];

% Set explicitly only if auto-detection fails.
digital_file = 'data_tradeoff_FD_SGPI_K8_Nt64_100ch_parfor_c2s.mat';
deep125_file = 'data_Fig2_DeepSSP_tradeoff_G1_2_4_8_N125_100ch.mat';


% SSP files are expected under this directory.
manual_ssp_dir = '';

pareto_prune_each_curve = true;

save_fig = false;
save_png = false;
save_eps = false;

%% -------------------- Paths --------------------
script_fullpath = mfilename('fullpath');

if isempty(script_fullpath)
    fig_dir = pwd;
else
    fig_dir = fileparts(script_fullpath);
end

project_root = fileparts(fig_dir);

% Style file.
if exist(fullfile(fig_dir,'plot_style.m'),'file')
    addpath(fig_dir);
elseif exist(fullfile(project_root,'plot_style.m'),'file')
    addpath(project_root);
elseif exist('plot_style','file') ~= 2
    error('Cannot find plot_style.m.');
end

Sstyle = plot_style('single');

%% -------------------- Locate SSP directory --------------------
if ~isempty(manual_ssp_dir)
    ssp_dir = manual_ssp_dir;
else
    candidates = { ...
        fullfile(fig_dir,'tradeoff_results_GC_parfor'), ...
        fullfile(project_root,'Fig2_tradeoff','tradeoff_results_GC_parfor'), ...
        fullfile(project_root,'tradeoff_results_GC_parfor')};

    ssp_dir = '';

    for ii = 1:numel(candidates)
        if exist(candidates{ii},'dir')
            ssp_dir = candidates{ii};
            break;
        end
    end
end

if isempty(ssp_dir)
    error(['Cannot find tradeoff_results_GC_parfor. ' ...
           'Set manual_ssp_dir at the top of the script.']);
end

%% -------------------- Locate Deep N=125 file --------------------
if isempty(deep125_file)

    deep_candidates = { ...
        fullfile(fig_dir,'results', ...
        'data_Fig2_DeepSSP_tradeoff_G1_2_4_8_N125_100ch.mat'), ...
        fullfile(project_root,'Fig2_tradeoff','results', ...
        'data_Fig2_DeepSSP_tradeoff_G1_2_4_8_N125_100ch.mat')};

    for ii = 1:numel(deep_candidates)
        if exist(deep_candidates{ii},'file')
            deep125_file = deep_candidates{ii};
            break;
        end
    end

    if isempty(deep125_file)
        hit = dir(fullfile(project_root,'**','*DeepSSP*tradeoff*N125*.mat'));

        if ~isempty(hit)
            deep125_file = fullfile(hit(1).folder,hit(1).name);
        end
    end
end

if isempty(deep125_file) || ~exist(deep125_file,'file')
    error(['Cannot find the Deep-SSP-HBF N=125 tradeoff result. ' ...
           'Set deep125_file manually.']);
end

%% -------------------- Locate digital file --------------------
if isempty(digital_file)
    digital_file = auto_find_digital_tradeoff(project_root,fig_dir);
end

if isempty(digital_file) || ~exist(digital_file,'file')
    error(['Cannot identify the fully digital tradeoff MAT file. ' ...
           'Set digital_file manually at the top of the script.']);
end

fprintf('Digital file:\n%s\n\n',digital_file);
fprintf('Deep N=125 file:\n%s\n\n',deep125_file);
fprintf('SSP directory:\n%s\n\n',ssp_dir);

%% ========================================================================
% Load digital benchmark
%% ========================================================================
Ddig = load(digital_file);

[CRB_dig,SR_dig] = extract_single_tradeoff(Ddig);

if pareto_prune_each_curve
    [CRB_dig,SR_dig] = pareto_front(CRB_dig,SR_dig);
else
    [CRB_dig,idx] = sort(CRB_dig,'ascend');
    SR_dig = SR_dig(idx);
end

%% ========================================================================
% Load SSP-HBF for G = 1,2,4,8
%% ========================================================================
SSP = cell(1,numel(G_list));

for iG = 1:numel(G_list)

    G = G_list(iG);

    file_ssp = fullfile(ssp_dir,sprintf( ...
        'data_tradeoff_GC_G%d_HBF_SGPI_K8_NRF8_100ch_parfor.mat',G));

    if ~exist(file_ssp,'file')
        % More flexible fallback.
        hit = dir(fullfile(ssp_dir,sprintf('*G%d*HBF*SGPI*.mat',G)));

        if isempty(hit)
            error('Cannot find SSP-HBF tradeoff file for G=%d.',G);
        end

        file_ssp = fullfile(hit(1).folder,hit(1).name);
    end

    T = load(file_ssp);
    [crb,sr] = extract_single_tradeoff(T);

    if pareto_prune_each_curve
        [crb,sr] = pareto_front(crb,sr);
    else
        [crb,idx] = sort(crb,'ascend');
        sr = sr(idx);
    end

    SSP{iG}.G = G;
    SSP{iG}.CRB = crb;
    SSP{iG}.SR = sr;
    SSP{iG}.file = file_ssp;
end

%% ========================================================================
% Load Deep-SSP-HBF, N = 125
%% ========================================================================
Ddeep = load(deep125_file);

[CRB_deep_raw,SR_deep_raw] = extract_multiG_tradeoff(Ddeep,G_list);

% Nominal training weight used for all frozen Deep-SSP-HBF models.
% Prefer the exact values stored in the MAT file; otherwise fall back to
% the nominal setting rho = delta_c/delta_s = 1e-3 (lambda = 1000).
[idx_train,training_text] = find_training_weight_index(Ddeep);

Deep = cell(1,numel(G_list));

for iG = 1:numel(G_list)

    crb_raw = CRB_deep_raw{iG};
    sr_raw  = SR_deep_raw{iG};

    if idx_train > numel(crb_raw) || idx_train > numel(sr_raw)
        error('Training-weight index exceeds the Deep tradeoff vector length.');
    end

    % Store the nominal training operating point BEFORE Pareto pruning.
    Deep{iG}.CRB_train = crb_raw(idx_train);
    Deep{iG}.SR_train  = sr_raw(idx_train);

    crb = crb_raw;
    sr  = sr_raw;

    if pareto_prune_each_curve
        [crb,sr] = pareto_front(crb,sr);
    else
        [crb,idx] = sort(crb,'ascend');
        sr = sr(idx);
    end

    Deep{iG}.G = G_list(iG);
    Deep{iG}.CRB = crb;
    Deep{iG}.SR = sr;
end

fprintf('Nominal Deep-SSP training weight: %s\n\n',training_text);

%% ========================================================================
% Colors and markers
%
% G=1 : color(1),  Deep counterpart color(11)
% G=2 : color(2),  Deep counterpart color(12)
% G=4 : color(3),  Deep counterpart color(13)
% G=8 : color(4),  Deep counterpart color(14)
%% ========================================================================
base_color_idx = [1 2 3 4];
light_color_idx = [11 12 13 14];

marker_list = {'o','s','^','d'};

C_ssp = zeros(4,3);
C_deep = zeros(4,3);

for iG = 1:4
    C_ssp(iG,:) = paper_color(base_color_idx(iG));
    C_deep(iG,:) = paper_color(light_color_idx(iG));
end

%% ========================================================================
% Plot
%% ========================================================================
fig = figure( ...
    'Color',Sstyle.background, ...
    'Units','inches', ...
    'Position',[1 1 Sstyle.fig_size], ...
    'Name','SR-CRLB tradeoff', ...
    'NumberTitle','off');

ax = axes(fig);
hold(ax,'on');

%% Digital benchmark
h_dig = plot(ax,CRB_dig,SR_dig, ...
    '-', ...
    'Color','k', ...
    'LineWidth',2.0, ...
    'Marker','p', ...
    'MarkerSize',6.5, ...
    'MarkerIndices',sparse_marker_indices(numel(CRB_dig),7));

%% SSP-HBF: solid / standard color
h_ssp = gobjects(1,4);

for iG = 1:4

    h_ssp(iG) = plot(ax, ...
        SSP{iG}.CRB,SSP{iG}.SR, ...
        '-', ...
        'Color',C_ssp(iG,:), ...
        'LineWidth',Sstyle.line_width, ...
        'Marker',marker_list{iG}, ...
        'MarkerSize',Sstyle.marker_size, ...
        'MarkerIndices',sparse_marker_indices(numel(SSP{iG}.CRB),7));
end

%% Deep-SSP-HBF: dashed / lightweight counterpart color
h_deep = gobjects(1,4);

for iG = 1:4

    h_deep(iG) = plot(ax, ...
        Deep{iG}.CRB,Deep{iG}.SR, ...
        '--', ...
        'Color',C_deep(iG,:), ...
        'LineWidth',Sstyle.line_width, ...
        'Marker',marker_list{iG}, ...
        'MarkerSize',Sstyle.marker_size, ...
        'MarkerIndices',sparse_marker_indices(numel(Deep{iG}.CRB),7));
end

%% Mark the nominal training weight on every Deep-SSP-HBF curve
%
% A black "x" is overlaid at the operating point corresponding to the
% training weight.  This makes generalization away from the nominal weight
% visually explicit without changing the G-specific color/marker encoding.
for iG = 1:4
    plot(ax, ...
        Deep{iG}.CRB_train,Deep{iG}.SR_train, ...
        'x', ...
        'Color','k', ...
        'LineWidth',1.45, ...
        'MarkerSize',7.5, ...
        'HandleVisibility','off');
end

xlabel(ax,'CRLB');
ylabel(ax,'Sum rate (nat/s/Hz)');

%% Compact factorized legend
%
% Avoid nine full curve labels.  Instead:
%   - marker/color identifies connectivity G;
%   - line style identifies SSP versus Deep-SSP;
%   - Digital keeps its own black star symbol.
%
% This is much lighter and matches the visual encoding of the figure.

hG = gobjects(1,4);

for iG = 1:4
    hG(iG) = plot(ax,nan,nan, ...
        'LineStyle','none', ...
        'Color',C_ssp(iG,:), ...
        'Marker',marker_list{iG}, ...
        'MarkerSize',Sstyle.marker_size, ...
        'LineWidth',Sstyle.line_width);
end

% Algorithm-style keys use neutral grayscale so that color remains reserved
% for the connectivity group.
h_key_ssp = plot(ax,nan,nan, ...
    '-', ...
    'Color',[0.25 0.25 0.25], ...
    'LineWidth',Sstyle.line_width);

h_key_deep = plot(ax,nan,nan, ...
    '--', ...
    'Color',[0.65 0.65 0.65], ...
    'LineWidth',Sstyle.line_width);

h_key_train = plot(ax,nan,nan, ...
    'x', ...
    'Color','k', ...
    'LineWidth',1.45, ...
    'MarkerSize',7.5);

h_leg = [h_dig,hG,h_key_ssp,h_key_deep,h_key_train];

leg_text = { ...
    'Digital', ...
    'G=1','G=2','G=4','G=8', ...
    'SSP-HBF','Deep-SSP-HBF','Training weight'};

lgd = legend(ax,h_leg,leg_text, ...
    'Location','southeast', ...
    'FontName',Sstyle.font_name, ...
    'FontSize',9.0, ...
    'Box','off', ...
    'Interpreter','none');

try
    lgd.NumColumns = 4;
catch
end

plot_style('single',ax);

%% ========================================================================
% Console summary
%% ========================================================================
fprintf('============================================================\n');
fprintf('Tradeoff plot loaded successfully\n');
fprintf('============================================================\n');
fprintf('Digital points: %d\n',numel(CRB_dig));

for iG = 1:4
    fprintf('G=%d | SSP points=%d | Deep N125 points=%d\n', ...
        G_list(iG),numel(SSP{iG}.CRB),numel(Deep{iG}.CRB));
end

fprintf('\nColor semantics:\n');
fprintf('  SSP-HBF      : color(1:4), solid\n');
fprintf('  Deep-SSP-HBF : color(11:14), dashed\n');
fprintf('  Same marker  : same connectivity G\n');
fprintf('  Black x      : nominal Deep-SSP training weight\n');
fprintf('  Training setting: %s\n',training_text);
fprintf('============================================================\n');

%% ========================================================================
% Optional export
%% ========================================================================
out_dir = fullfile(fig_dir,'results','figures_final_tradeoff');

if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

if save_fig
    savefig(fig,fullfile(out_dir,'Fig3_SR_CRLB_tradeoff.fig'));
end

if save_png
    print(fig,fullfile(out_dir,'Fig3_SR_CRLB_tradeoff.png'), ...
        '-dpng',sprintf('-r%d',Sstyle.export_dpi));
end

if save_eps
    print(fig,fullfile(out_dir,'Fig3_SR_CRLB_tradeoff.eps'), ...
        '-depsc2','-painters');
end

%% ========================================================================
% Local functions
%% ========================================================================

function c = paper_color(idx)
%PAPER_COLOR Use user's color(idx) function when available.
%
% Expected convention:
%   color(1:4)   = main colors
%   color(11:14) = lightweight versions

    if exist('color','file') == 2
        try
            c = feval('color',idx);
            c = double(c(:).');

            if numel(c)==3 && all(isfinite(c))
                return;
            end
        catch
        end
    end

    % Fallback palette if user's color.m is not on the path.
    base = [ ...
        0.0000 0.4470 0.7410; ...
        0.8500 0.3250 0.0980; ...
        0.0000 0.7000 0.7000; ...
        0.4940 0.1840 0.5560];

    if idx>=1 && idx<=4
        c = base(idx,:);
        return;
    end

    if idx>=11 && idx<=14
        c0 = base(idx-10,:);
        lighten = 0.48;
        c = c0 + lighten*(1-c0);
        return;
    end

    error('Unsupported fallback color index %d.',idx);
end


function idx = sparse_marker_indices(N,nmarker)
%SPARSE_MARKER_INDICES Roughly nmarker markers over the full curve.

    if N<=nmarker
        idx = 1:N;
    else
        idx = unique(round(linspace(1,N,nmarker)));
    end
end


function [CRB,SR] = extract_single_tradeoff(T)
%EXTRACT_SINGLE_TRADEOFF Read one SR-CRLB curve from a MAT structure.

    if isfield(T,'CRB_front') && isfield(T,'SR_front')
        CRB = double(T.CRB_front(:).');
        SR  = double(T.SR_front(:).');
        return;
    end

    if isfield(T,'CRB_avg')
        CRB = double(T.CRB_avg(:).');
    elseif isfield(T,'CRB_all')
        CRB = mean(double(T.CRB_all),1,'omitnan');
    else
        CRB = find_numeric_field(T,'crb');
    end

    if isfield(T,'SR_avg')
        SR = double(T.SR_avg(:).');
    elseif isfield(T,'SR_all')
        SR = mean(double(T.SR_all),1,'omitnan');
    else
        SR = find_numeric_field(T,'sr');
    end

    CRB = double(CRB(:).');
    SR  = double(SR(:).');

    valid = isfinite(CRB) & isfinite(SR);
    CRB = CRB(valid);
    SR = SR(valid);

    if isempty(CRB) || numel(CRB)~=numel(SR)
        error('Could not extract a valid SR-CRLB tradeoff curve.');
    end
end


function x = find_numeric_field(T,token)
%FIND_NUMERIC_FIELD Fallback for simple MAT files.

    fields = fieldnames(T);
    x = [];

    for k = 1:numel(fields)

        f = fields{k};

        if contains(lower(f),lower(token))
            v = T.(f);

            if isnumeric(v) && isvector(v)
                x = double(v(:).');
                return;
            end
        end
    end

    error('Could not find a numeric field containing "%s".',token);
end


function [CRB_cell,SR_cell] = extract_multiG_tradeoff(T,G_target)
%EXTRACT_MULTIG_TRADEOFF Extract Deep N=125 tradeoff curves for all G.
%
% Supported layouts include:
%
%   G_list  = [1 2 4 8]
%   SR_avg  = 4 x Nw
%   CRLB_avg = 4 x Nw
%
% which is exactly the layout produced by
% data_Fig2_DeepSSP_tradeoff_G1_2_4_8_N125_100ch.mat.
%
% It also supports transposed Nw x 4 arrays and cell/struct layouts.

    CRB_cell = cell(1,numel(G_target));
    SR_cell  = cell(1,numel(G_target));

    %% ================================================================
    % Case 1: explicit G_list + SR_avg + CRLB_avg
    % ================================================================
    if isfield(T,'G_list') && isfield(T,'SR_avg') && isfield(T,'CRLB_avg')

        G_file = double(T.G_list(:).');
        SRmat  = double(T.SR_avg);
        CRBmat = double(T.CRLB_avg);

        % Orient as nG x Nw.
        if size(SRmat,1) ~= numel(G_file) && size(SRmat,2) == numel(G_file)
            SRmat = SRmat.';
        end

        if size(CRBmat,1) ~= numel(G_file) && size(CRBmat,2) == numel(G_file)
            CRBmat = CRBmat.';
        end

        if size(SRmat,1) ~= numel(G_file) || ...
           size(CRBmat,1) ~= numel(G_file) || ...
           size(SRmat,2) ~= size(CRBmat,2)

            error(['Deep N=125 file has G_list/SR_avg/CRLB_avg, but their ' ...
                   'dimensions are inconsistent.']);
        end

        for iG = 1:numel(G_target)

            row = find(G_file == G_target(iG),1);

            if isempty(row)
                error('Deep N=125 file does not contain G=%d.',G_target(iG));
            end

            SR_cell{iG}  = SRmat(row,:);
            CRB_cell{iG} = CRBmat(row,:);
        end

        return;
    end

    %% ================================================================
    % Case 2: explicit G_list + SR_all + CRLB_all
    %
    % Expected dimensions are typically:
    %   nG x Nchannel x Nweight
    % ================================================================
    if isfield(T,'G_list') && isfield(T,'SR_all') && isfield(T,'CRLB_all')

        G_file = double(T.G_list(:).');
        SRall  = double(T.SR_all);
        CRBall = double(T.CRLB_all);

        if ndims(SRall) == 3 && ndims(CRBall) == 3 && ...
                size(SRall,1) == numel(G_file) && ...
                size(CRBall,1) == numel(G_file)

            SRmat  = squeeze(mean(SRall,2,'omitnan'));
            CRBmat = squeeze(mean(CRBall,2,'omitnan'));

            for iG = 1:numel(G_target)

                row = find(G_file == G_target(iG),1);

                if isempty(row)
                    error('Deep N=125 file does not contain G=%d.',G_target(iG));
                end

                SR_cell{iG}  = SRmat(row,:);
                CRB_cell{iG} = CRBmat(row,:);
            end

            return;
        end
    end

    %% ================================================================
    % Case 3: generic numeric matrices / cells
    % ================================================================
    fields = fieldnames(T);

    sr_name  = '';
    crb_name = '';

    for k = 1:numel(fields)

        name = lower(fields{k});
        value = T.(fields{k});

        if isempty(sr_name) && contains(name,'sr') && ...
                (isnumeric(value) || iscell(value))
            sr_name = fields{k};
        end

        if isempty(crb_name) && contains(name,'crb') && ...
                (isnumeric(value) || iscell(value))
            crb_name = fields{k};
        end
    end

    if ~isempty(sr_name) && ~isempty(crb_name)

        SRraw  = T.(sr_name);
        CRBraw = T.(crb_name);

        if iscell(SRraw) && iscell(CRBraw) && ...
                numel(SRraw) >= numel(G_target) && ...
                numel(CRBraw) >= numel(G_target)

            for iG = 1:numel(G_target)
                SR_cell{iG}  = double(SRraw{iG}(:).');
                CRB_cell{iG} = double(CRBraw{iG}(:).');
            end

            return;
        end

        if isnumeric(SRraw) && isnumeric(CRBraw)

            [SRmat,ok1]   = orient_multiG_matrix(double(SRraw),numel(G_target));
            [CRBmat,ok2]  = orient_multiG_matrix(double(CRBraw),numel(G_target));

            if ok1 && ok2 && isequal(size(SRmat),size(CRBmat))

                for iG = 1:numel(G_target)
                    SR_cell{iG}  = SRmat(iG,:);
                    CRB_cell{iG} = CRBmat(iG,:);
                end

                return;
            end
        end
    end

    %% ================================================================
    % Case 4: nested structs named G1/G2/G4/G8
    % ================================================================
    for iG = 1:numel(G_target)

        G = G_target(iG);

        candidate_names = { ...
            sprintf('G%d',G), ...
            sprintf('g%d',G), ...
            sprintf('Deep_G%d',G), ...
            sprintf('Deep125_G%d',G)};

        found = false;

        for kk = 1:numel(candidate_names)

            if isfield(T,candidate_names{kk})

                A = T.(candidate_names{kk});

                if isstruct(A)
                    [CRB_cell{iG},SR_cell{iG}] = extract_single_tradeoff(A);
                    found = true;
                    break;
                end
            end
        end

        if ~found
            break;
        end
    end

    if all(~cellfun(@isempty,CRB_cell)) && ...
       all(~cellfun(@isempty,SR_cell))
        return;
    end

    fprintf('\nFields available in Deep N=125 file:\n');
    disp(fields);

    error(['Could not automatically decode the Deep N=125 MAT file. ' ...
           'Expected G_list with SR_avg/CRLB_avg or an equivalent layout.']);
end


function [A,ok] = orient_multiG_matrix(A,nG)
%ORIENT_MULTIG_MATRIX Return nG-by-N representation.

    ok = false;

    if isvector(A)
        return;
    end

    if size(A,1)==nG
        ok = true;
        return;
    end

    if size(A,2)==nG
        A = A.';
        ok = true;
    end
end


function [CRB_front,SR_front] = pareto_front(CRB,SR)
%PARETO_FRONT Remove dominated points and sort by increasing CRLB.
%
% Desired direction:
%   smaller CRLB is better
%   larger SR is better

    CRB = double(CRB(:).');
    SR = double(SR(:).');

    valid = isfinite(CRB) & isfinite(SR);
    CRB = CRB(valid);
    SR = SR(valid);

    N = numel(CRB);
    dominated = false(1,N);

    tol_crb = 1e-12;
    tol_sr = 1e-10;

    for i = 1:N

        for j = 1:N

            if i==j
                continue;
            end

            no_worse_crb = CRB(j) <= CRB(i) + tol_crb;
            no_worse_sr  = SR(j)  >= SR(i)  - tol_sr;

            better_crb = CRB(j) < CRB(i) - tol_crb;
            better_sr  = SR(j)  > SR(i)  + tol_sr;

            if no_worse_crb && no_worse_sr && (better_crb || better_sr)
                dominated(i) = true;
                break;
            end
        end
    end

    CRB_front = CRB(~dominated);
    SR_front = SR(~dominated);

    [CRB_front,idx] = sort(CRB_front,'ascend');
    SR_front = SR_front(idx);
end


function [idx_train,txt] = find_training_weight_index(T)
%FIND_TRAINING_WEIGHT_INDEX Locate the nominal Deep training weight.
%
% Preferred metadata:
%   lambda_train with lambda_list
%   rho_train    with rho_list
%
% Fallback:
%   rho = delta_c/delta_s = 1e-3
%   lambda = delta_s/delta_c = 1000

    rho_nominal = 1e-3;
    lambda_nominal = 1/rho_nominal;

    %% Preferred: lambda metadata
    if isfield(T,'lambda_list')

        lambda_list = to_numeric_row(T.lambda_list);

        if ~isempty(lambda_list)

            if isfield(T,'lambda_train')
                lambda_train = to_scalar_numeric(T.lambda_train);
            else
                lambda_train = lambda_nominal;
            end

            if ~isempty(lambda_train) && isfinite(lambda_train)
                [~,idx_train] = min(abs(log10(max(lambda_list,realmin)) - ...
                                        log10(max(lambda_train,realmin))));
                txt = sprintf('lambda = %.4g',lambda_train);
                return;
            end
        end
    end

    %% Second choice: rho metadata
    if isfield(T,'rho_list')

        rho_list = to_numeric_row(T.rho_list);

        if ~isempty(rho_list)

            if isfield(T,'rho_train')
                rho_train = to_scalar_numeric(T.rho_train);
            else
                rho_train = rho_nominal;
            end

            if ~isempty(rho_train) && isfinite(rho_train)
                [~,idx_train] = min(abs(log10(max(rho_list,realmin)) - ...
                                        log10(max(rho_train,realmin))));
                txt = sprintf('rho = %.4g',rho_train);
                return;
            end
        end
    end

    %% Third choice: derive rho from delta_c/delta_s
    if isfield(T,'delta_c_list') && isfield(T,'delta_s_list')

        dc = to_numeric_row(T.delta_c_list);
        ds = to_numeric_row(T.delta_s_list);

        if numel(dc)==numel(ds) && ~isempty(dc)
            rho_list = dc ./ max(ds,realmin);
            [~,idx_train] = min(abs(log10(max(rho_list,realmin)) - ...
                                    log10(rho_nominal)));
            txt = sprintf('rho = %.4g',rho_nominal);
            return;
        end
    end

    error(['Cannot identify the nominal training-weight index. ' ...
           'Expected lambda_list/rho_list or delta_c_list/delta_s_list.']);
end


function x = to_numeric_row(v)
%TO_NUMERIC_ROW Convert numeric/string/cell weight vectors to doubles.

    if isnumeric(v)
        x = double(v(:).');
        return;
    end

    if isstring(v) || ischar(v)
        x = str2double(string(v));
        x = double(x(:).');
        x = x(isfinite(x));
        return;
    end

    if iscell(v)
        x = nan(1,numel(v));

        for k = 1:numel(v)
            if isnumeric(v{k}) && isscalar(v{k})
                x(k) = double(v{k});
            else
                x(k) = str2double(string(v{k}));
            end
        end

        return;
    end

    x = [];
end


function x = to_scalar_numeric(v)
%TO_SCALAR_NUMERIC Convert a stored scalar weight to double.

    x = [];

    if isnumeric(v) && isscalar(v)
        x = double(v);
    elseif isstring(v) || ischar(v)
        tmp = str2double(string(v));
        if isscalar(tmp)
            x = double(tmp);
        end
    elseif iscell(v) && numel(v)==1
        x = to_scalar_numeric(v{1});
    end
end


function file = auto_find_digital_tradeoff(project_root,fig_dir)
%AUTO_FIND_DIGITAL_TRADEOFF Search likely fully-digital result files.

    roots = { ...
        fullfile(fig_dir,'results'), ...
        fig_dir, ...
        fullfile(project_root,'Fig2_tradeoff'), ...
        project_root};

    files = {};

    patterns = { ...
        '*digital*tradeoff*.mat', ...
        '*tradeoff*digital*.mat', ...
        '*fully*digital*.mat', ...
        '*FD*tradeoff*.mat', ...
        'tradeoff_summary*.mat'};

    for rr = 1:numel(roots)

        if ~exist(roots{rr},'dir')
            continue;
        end

        for pp = 1:numel(patterns)

            if strcmp(roots{rr},project_root)
                hit = dir(fullfile(roots{rr},'**',patterns{pp}));
            else
                hit = dir(fullfile(roots{rr},patterns{pp}));
            end

            for k = 1:numel(hit)
                files{end+1} = fullfile(hit(k).folder,hit(k).name); %#ok<AGROW>
            end
        end
    end

    files = unique(files,'stable');
    file = '';

    % Select first file containing some plausible SR and CRB arrays.
    for k = 1:numel(files)

        try
            info = whos('-file',files{k});
            names = lower(string({info.name}));

            has_sr = any(contains(names,'sr'));
            has_crb = any(contains(names,'crb'));

            if has_sr && has_crb
                file = files{k};
                return;
            end
        catch
        end
    end
end
