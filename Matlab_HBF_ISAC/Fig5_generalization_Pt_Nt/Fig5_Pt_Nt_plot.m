clc
clear
close all

%% ========================================================================
% Pt/Nt generalization: final system-level performance plots
%
% Two separate MATLAB figures are generated for vertical stacking:
%   Fig. (a): performance versus transmit power
%   Fig. (b): performance versus number of transmit antennas
%
% Comparison:
%   SSP-HBF long-run reference
%   Deep-SSP-HBF with selected practical depth N = 125
%
% Fixed architecture:
%   G = 4
%
% Visual encoding:
%   Left y-axis  / Sum rate : blue family
%   Right y-axis / CRLB     : red/orange family
%   SSP-HBF                 : darker color, solid line
%   Deep-SSP-HBF            : lightweight counterpart, dashed line
%   Black crosses           : nominal training configuration
%
% Nominal training configuration:
%   Pt = 20 dBm, Nt = 64
%
% This script follows the same single-column style as the convergence
% and K/M generalization figures.
%% ========================================================================

%% -------------------- User options --------------------
base_file = '';
supp_file = '';

save_fig = false;
save_png = false;
save_eps = false;

% Visual separation for the right-axis CRLB curves.
% These values change ONLY the displayed axis limits, not the CRLB data.
% Larger top padding moves CRLB curves downward in the panel.
crlb_pt_top_padding = 0.22;
crlb_pt_bottom_padding = 0.05;

crlb_nt_top_padding = 0.22;
crlb_nt_bottom_padding = 0.05;

%% -------------------- Paths --------------------
script_fullpath = mfilename('fullpath');

if isempty(script_fullpath)
    fig_dir = pwd;
else
    fig_dir = fileparts(script_fullpath);
end

project_root = fileparts(fig_dir);

% Unified paper style.
if exist(fullfile(fig_dir,'plot_style.m'),'file')
    addpath(fig_dir);
elseif exist(fullfile(project_root,'plot_style.m'),'file')
    addpath(project_root);
elseif exist('plot_style','file') ~= 2
    error(['Cannot find plot_style.m. Put it in this folder, the project ' ...
           'root, or on the MATLAB path.']);
end

S = plot_style('single');

%% -------------------- Locate result files --------------------
base_name = 'data_Fig5_DeepSSP_generalization_Pt_Nt_G4_100ch.mat';
supp_name = 'data_Fig5_DeepSSP_generalization_Pt_Nt_G4_N125_supplement_100ch.mat';

if isempty(base_file)
    base_file = locate_result_file(fig_dir,project_root,base_name);
end

if isempty(supp_file)
    supp_file = locate_result_file(fig_dir,project_root,supp_name);
end

fprintf('Base result:\n%s\n\n',base_file);
fprintf('N=125 supplement:\n%s\n\n',supp_file);

B = load(base_file);
D = load(supp_file);

%% -------------------- Sweep vectors --------------------
Pt_dBm_list = get_vector(B, ...
    {'Pt_dBm_list','Pt_list_dBm','Pt_list','Pt_dBm'}, ...
    [0 5 10 15 20 25]);

Nt_list = get_vector(B, ...
    {'Nt_list','Ntx_list','Nt_values'}, ...
    [32 48 64 80 96 112 128]);

% Check supplement sweep vectors when available.
Pt_dBm_list_D = get_vector_optional(D, ...
    {'Pt_dBm_list','Pt_list_dBm','Pt_list','Pt_dBm'});

Nt_list_D = get_vector_optional(D, ...
    {'Nt_list','Ntx_list','Nt_values'});

if ~isempty(Pt_dBm_list_D) && ~isequal(Pt_dBm_list,Pt_dBm_list_D)
    error('Transmit-power sweep mismatch between base and supplement.');
end

if ~isempty(Nt_list_D) && ~isequal(Nt_list,Nt_list_D)
    error('Transmit-antenna sweep mismatch between base and supplement.');
end

%% -------------------- Extract result structures --------------------
Pt_ref_struct = get_sweep_struct(B, ...
    {'Ptres','PtRes','Pt_result','PtResult','Pt_results'}, ...
    numel(Pt_dBm_list));

Nt_ref_struct = get_sweep_struct(B, ...
    {'Ntres','NtRes','Nt_result','NtResult','Nt_results'}, ...
    numel(Nt_list));

Pt_125_struct = get_sweep_struct(D, ...
    {'Pt125','Pt_125','Ptres125','PtRes125','Pt_result','PtResult','Pt_results'}, ...
    numel(Pt_dBm_list));

Nt_125_struct = get_sweep_struct(D, ...
    {'Nt125','Nt_125','Ntres125','NtRes125','Nt_result','NtResult','Nt_results'}, ...
    numel(Nt_list));

%% -------------------- Extract averaged physical metrics --------------------
% Long-run SSP-HBF reference from the base file.
SR_pt_ref = extract_metric_average(Pt_ref_struct, ...
    {'SRRef_avg','SR_ref_avg','SRSSP_avg','SR_SSP_avg','SRref_avg'}, ...
    {'SRRef','SR_ref','SRSSP','SR_SSP','SRref'}, ...
    numel(Pt_dBm_list));

CRLB_pt_ref = extract_metric_average(Pt_ref_struct, ...
    {'CRLBRef_avg','CRLB_ref_avg','CRLBSSP_avg','CRLB_SSP_avg','CRLBref_avg'}, ...
    {'CRLBRef','CRLB_ref','CRLBSSP','CRLB_SSP','CRLBref'}, ...
    numel(Pt_dBm_list));

SR_nt_ref = extract_metric_average(Nt_ref_struct, ...
    {'SRRef_avg','SR_ref_avg','SRSSP_avg','SR_SSP_avg','SRref_avg'}, ...
    {'SRRef','SR_ref','SRSSP','SR_SSP','SRref'}, ...
    numel(Nt_list));

CRLB_nt_ref = extract_metric_average(Nt_ref_struct, ...
    {'CRLBRef_avg','CRLB_ref_avg','CRLBSSP_avg','CRLB_SSP_avg','CRLBref_avg'}, ...
    {'CRLBRef','CRLB_ref','CRLBSSP','CRLB_SSP','CRLBref'}, ...
    numel(Nt_list));

% Deep-SSP-HBF with N=125 from the supplement.
SR_pt_125 = extract_metric_average(Pt_125_struct, ...
    {'SR_avg','SR125_avg','SR_125_avg','SRDeep_avg','SR_Deep_avg'}, ...
    {'SR','SR125','SR_125','SRDeep','SR_Deep'}, ...
    numel(Pt_dBm_list));

CRLB_pt_125 = extract_metric_average(Pt_125_struct, ...
    {'CRLB_avg','CRLB125_avg','CRLB_125_avg','CRLBDeep_avg','CRLB_Deep_avg'}, ...
    {'CRLB','CRLB125','CRLB_125','CRLBDeep','CRLB_Deep'}, ...
    numel(Pt_dBm_list));

SR_nt_125 = extract_metric_average(Nt_125_struct, ...
    {'SR_avg','SR125_avg','SR_125_avg','SRDeep_avg','SR_Deep_avg'}, ...
    {'SR','SR125','SR_125','SRDeep','SR_Deep'}, ...
    numel(Nt_list));

CRLB_nt_125 = extract_metric_average(Nt_125_struct, ...
    {'CRLB_avg','CRLB125_avg','CRLB_125_avg','CRLBDeep_avg','CRLB_Deep_avg'}, ...
    {'CRLB','CRLB125','CRLB_125','CRLBDeep','CRLB_Deep'}, ...
    numel(Nt_list));

%% -------------------- Nominal training setup --------------------
Pt_train_dBm = get_scalar(B, ...
    {'Pt0_dBm','Pt_train_dBm','Pt_dBm0','Pt_nominal_dBm'},20);

Nt_train = get_scalar(B, ...
    {'Nt0','Nt_train','Nt_nominal'},64);

G = get_scalar(B,{'G'},4);
Nexe = get_scalar(D,{'Nexe','N_exe','N125'},125);

fprintf('Detected/used settings:\n');
fprintf('  G                 = %g\n',G);
fprintf('  Deep execution N  = %g\n',Nexe);
fprintf('  Training Pt       = %g dBm\n',Pt_train_dBm);
fprintf('  Training Nt       = %g\n\n',Nt_train);

%% ========================================================================
% Colors
%
% Standard dual-axis convention:
%   left / sum rate : blue
%   right / CRLB    : red/orange
%
% User's color.m convention:
%   Sum rate SSP   : color(1)
%   Sum rate Deep  : color(11)
%   CRLB SSP       : color(2)
%   CRLB Deep      : color(12)
%% ========================================================================
c_sr_ssp    = paper_color(1);
c_sr_deep   = paper_color(11);

c_crlb_ssp  = paper_color(2);
c_crlb_deep = paper_color(12);

marker_sr   = 'o';
marker_crlb = 's';

%% ========================================================================
% FIGURE A: Performance versus transmit power
%% ========================================================================
figPt = figure( ...
    'Color',S.background, ...
    'Units','inches', ...
    'Position',[1 1 S.fig_size], ...
    'Name','Generalization versus transmit power', ...
    'NumberTitle','off');

axPt = axes(figPt);
hold(axPt,'on');

%% Sum rate -- left axis
yyaxis(axPt,'left');

hPt_sr_ssp = plot(axPt, ...
    Pt_dBm_list,SR_pt_ref, ...
    '-', ...
    'Color',c_sr_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_sr, ...
    'MarkerSize',S.marker_size);

hPt_sr_deep = plot(axPt, ...
    Pt_dBm_list,SR_pt_125, ...
    '--', ...
    'Color',c_sr_deep, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_sr, ...
    'MarkerSize',S.marker_size);

ylabel(axPt,'Sum rate (nat/s/Hz)', ...
    'Interpreter','none');

idx_pt_train = find(abs(Pt_dBm_list-Pt_train_dBm)<1e-12,1);

if ~isempty(idx_pt_train)
    plot(axPt,Pt_train_dBm,SR_pt_125(idx_pt_train), ...
        'x','Color','k','LineWidth',1.4,'MarkerSize',7.2, ...
        'HandleVisibility','off');
end

%% CRLB -- right axis
yyaxis(axPt,'right');

hPt_crlb_ssp = plot(axPt, ...
    Pt_dBm_list,CRLB_pt_ref, ...
    '-', ...
    'Color',c_crlb_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_crlb, ...
    'MarkerSize',S.marker_size);

hPt_crlb_deep = plot(axPt, ...
    Pt_dBm_list,CRLB_pt_125, ...
    '--', ...
    'Color',c_crlb_deep, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_crlb, ...
    'MarkerSize',S.marker_size);

ylabel(axPt,'CRLB', ...
    'Interpreter','none');

if ~isempty(idx_pt_train)
    plot(axPt,Pt_train_dBm,CRLB_pt_125(idx_pt_train), ...
        'x','Color','k','LineWidth',1.4,'MarkerSize',7.2, ...
        'HandleVisibility','off');
end

set_metric_axis_padding(axPt, ...
    [CRLB_pt_ref(:); CRLB_pt_125(:)], ...
    crlb_pt_bottom_padding,crlb_pt_top_padding);

%% Common x-axis and styling
xlabel(axPt,'Transmit power (dBm)', ...
    'Interpreter','none');

xticks(axPt,Pt_dBm_list);
xlim(axPt,[min(Pt_dBm_list) max(Pt_dBm_list)]);

axPt.YAxis(1).Color = c_sr_ssp;
axPt.YAxis(2).Color = c_crlb_ssp;

h_train_Pt = plot(axPt,nan,nan, ...
    'x','Color','k','LineWidth',1.4,'MarkerSize',7.2);

legend(axPt, ...
    [hPt_sr_ssp hPt_sr_deep hPt_crlb_ssp hPt_crlb_deep h_train_Pt], ...
    {'Sum rate, SSP-HBF', ...
     'Sum rate, Deep-SSP-HBF', ...
     'CRLB, SSP-HBF', ...
     'CRLB, Deep-SSP-HBF', ...
     'Training setting'}, ...
    'Location','best', ...
    'FontName',S.font_name, ...
    'FontSize',S.legend_font, ...
    'Box','off', ...
    'Interpreter','none');

plot_style('single',axPt);

%% ========================================================================
% FIGURE B: Performance versus number of transmit antennas
%% ========================================================================
figNt = figure( ...
    'Color',S.background, ...
    'Units','inches', ...
    'Position',[6.8 1 S.fig_size], ...
    'Name','Generalization versus number of transmit antennas', ...
    'NumberTitle','off');

axNt = axes(figNt);
hold(axNt,'on');

%% Sum rate -- left axis
yyaxis(axNt,'left');

hNt_sr_ssp = plot(axNt, ...
    Nt_list,SR_nt_ref, ...
    '-', ...
    'Color',c_sr_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_sr, ...
    'MarkerSize',S.marker_size);

hNt_sr_deep = plot(axNt, ...
    Nt_list,SR_nt_125, ...
    '--', ...
    'Color',c_sr_deep, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_sr, ...
    'MarkerSize',S.marker_size);

ylabel(axNt,'Sum rate (nat/s/Hz)', ...
    'Interpreter','none');

idx_nt_train = find(Nt_list==Nt_train,1);

if ~isempty(idx_nt_train)
    plot(axNt,Nt_train,SR_nt_125(idx_nt_train), ...
        'x','Color','k','LineWidth',1.4,'MarkerSize',7.2, ...
        'HandleVisibility','off');
end

%% CRLB -- right axis
yyaxis(axNt,'right');

hNt_crlb_ssp = plot(axNt, ...
    Nt_list,CRLB_nt_ref, ...
    '-', ...
    'Color',c_crlb_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_crlb, ...
    'MarkerSize',S.marker_size);

hNt_crlb_deep = plot(axNt, ...
    Nt_list,CRLB_nt_125, ...
    '--', ...
    'Color',c_crlb_deep, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_crlb, ...
    'MarkerSize',S.marker_size);

ylabel(axNt,'CRLB', ...
    'Interpreter','none');

if ~isempty(idx_nt_train)
    plot(axNt,Nt_train,CRLB_nt_125(idx_nt_train), ...
        'x','Color','k','LineWidth',1.4,'MarkerSize',7.2, ...
        'HandleVisibility','off');
end

set_metric_axis_padding(axNt, ...
    [CRLB_nt_ref(:); CRLB_nt_125(:)], ...
    crlb_nt_bottom_padding,crlb_nt_top_padding);

%% Common x-axis and styling
xlabel(axNt,'Number of transmit antennas', ...
    'Interpreter','none');

xticks(axNt,Nt_list);
xlim(axNt,[min(Nt_list) max(Nt_list)]);

axNt.YAxis(1).Color = c_sr_ssp;
axNt.YAxis(2).Color = c_crlb_ssp;

h_train_Nt = plot(axNt,nan,nan, ...
    'x','Color','k','LineWidth',1.4,'MarkerSize',7.2);

legend(axNt, ...
    [hNt_sr_ssp hNt_sr_deep hNt_crlb_ssp hNt_crlb_deep h_train_Nt], ...
    {'Sum rate, SSP-HBF', ...
     'Sum rate, Deep-SSP-HBF', ...
     'CRLB, SSP-HBF', ...
     'CRLB, Deep-SSP-HBF', ...
     'Training setting'}, ...
    'Location','best', ...
    'FontName',S.font_name, ...
    'FontSize',S.legend_font, ...
    'Box','off', ...
    'Interpreter','none');

plot_style('single',axNt);

%% ========================================================================
% Console summary
%% ========================================================================
fprintf('============================================================\n');
fprintf('Pt/Nt system-level generalization\n');
fprintf('============================================================\n');
fprintf('Connectivity           : G = %g\n',G);
fprintf('Deep execution depth   : %g\n',Nexe);
fprintf('Training configuration : Pt = %g dBm, Nt = %g\n', ...
    Pt_train_dBm,Nt_train);

fprintf('\nTransmit-power sweep:\n');
fprintf('Pt(dBm) | SR ref | SR Deep | CRLB ref | CRLB Deep\n');

for i = 1:numel(Pt_dBm_list)
    fprintf('%7.1f | %7.4f | %7.4f | %8.4e | %8.4e\n', ...
        Pt_dBm_list(i), ...
        SR_pt_ref(i),SR_pt_125(i), ...
        CRLB_pt_ref(i),CRLB_pt_125(i));
end

fprintf('\nTransmit-antenna sweep:\n');
fprintf('Nt | SR ref | SR Deep | CRLB ref | CRLB Deep\n');

for i = 1:numel(Nt_list)
    fprintf('%3d | %7.4f | %7.4f | %8.4e | %8.4e\n', ...
        Nt_list(i), ...
        SR_nt_ref(i),SR_nt_125(i), ...
        CRLB_nt_ref(i),CRLB_nt_125(i));
end

fprintf('============================================================\n');

%% ========================================================================
% Optional export
%% ========================================================================
out_dir = fullfile(fig_dir,'results','figures_final_Pt_Nt');

if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

if save_fig
    savefig(figPt,fullfile(out_dir,'Fig5a_performance_vs_Pt.fig'));
    savefig(figNt,fullfile(out_dir,'Fig5b_performance_vs_Nt.fig'));
end

if save_png
    print(figPt,fullfile(out_dir,'Fig5a_performance_vs_Pt.png'), ...
        '-dpng',sprintf('-r%d',S.export_dpi));

    print(figNt,fullfile(out_dir,'Fig5b_performance_vs_Nt.png'), ...
        '-dpng',sprintf('-r%d',S.export_dpi));
end

if save_eps
    print(figPt,fullfile(out_dir,'Fig5a_performance_vs_Pt.eps'), ...
        '-depsc2','-painters');

    print(figNt,fullfile(out_dir,'Fig5b_performance_vs_Nt.eps'), ...
        '-depsc2','-painters');
end

%% ========================================================================
% Local helpers
%% ========================================================================
function file = locate_result_file(fig_dir,project_root,name)

    candidates = { ...
        fullfile(fig_dir,'results',name), ...
        fullfile(fig_dir,name), ...
        fullfile(project_root,'Fig5_generalization_Pt_Nt','results',name), ...
        fullfile(project_root,'results',name)};

    file = '';

    for i = 1:numel(candidates)
        if exist(candidates{i},'file')
            file = candidates{i};
            return;
        end
    end

    hit = dir(fullfile(project_root,'**',name));

    if ~isempty(hit)
        file = fullfile(hit(1).folder,hit(1).name);
        return;
    end

    error('Cannot find result file:\n%s',name);
end


function x = get_vector(S,names,fallback)

    x = get_vector_optional(S,names);

    if isempty(x)
        x = fallback;
        warning('Using fallback sweep vector: [%s].',num2str(x));
    end
end


function x = get_vector_optional(S,names)

    x = [];

    for i = 1:numel(names)
        if isfield(S,names{i}) && isnumeric(S.(names{i}))
            v = double(S.(names{i}));
            if isvector(v)
                x = v(:).';
                return;
            end
        end
    end
end


function x = get_scalar(S,names,fallback)

    x = fallback;

    for i = 1:numel(names)
        if isfield(S,names{i}) && isnumeric(S.(names{i})) && isscalar(S.(names{i}))
            x = double(S.(names{i}));
            return;
        end
    end
end


function R = get_sweep_struct(S,preferred_names,nSweep)
%GET_SWEEP_STRUCT Find the result struct corresponding to one sweep.

    for i = 1:numel(preferred_names)
        if isfield(S,preferred_names{i}) && isstruct(S.(preferred_names{i}))
            R = S.(preferred_names{i});
            return;
        end
    end

    % Fallback: search all top-level structs for a numeric field whose
    % first or second dimension matches the number of sweep points.
    fn = fieldnames(S);

    for i = 1:numel(fn)

        X = S.(fn{i});

        if ~isstruct(X)
            continue;
        end

        subfn = fieldnames(X);

        for j = 1:numel(subfn)

            V = X.(subfn{j});

            if isnumeric(V) && ~isscalar(V)
                sz = size(V);

                if any(sz==nSweep)
                    R = X;
                    warning('Automatically using struct "%s" for a %d-point sweep.', ...
                        fn{i},nSweep);
                    return;
                end
            end
        end
    end

    error(['Cannot identify the result struct for the %d-point sweep.\n' ...
           'Top-level fields are:\n%s'], ...
           nSweep,strjoin(fieldnames(S),', '));
end


function y = extract_metric_average(R,avg_names,raw_names,nSweep)
%EXTRACT_METRIC_AVERAGE Extract one averaged metric from a sweep struct.

    % First prefer explicitly saved averages.
    for i = 1:numel(avg_names)

        if isfield(R,avg_names{i}) && isnumeric(R.(avg_names{i}))

            v = double(R.(avg_names{i}));

            if isvector(v) && numel(v)==nSweep
                y = v(:).';
                return;
            end
        end
    end

    % Otherwise average the raw channel-wise matrix over the channel dim.
    for i = 1:numel(raw_names)

        if ~isfield(R,raw_names{i}) || ~isnumeric(R.(raw_names{i}))
            continue;
        end

        X = double(R.(raw_names{i}));

        if isvector(X) && numel(X)==nSweep
            y = X(:).';
            return;
        end

        if ndims(X)==2

            if size(X,1)==nSweep
                y = mean(X,2,'omitnan').';
                return;
            elseif size(X,2)==nSweep
                y = mean(X,1,'omitnan');
                return;
            end
        end
    end

    error(['Cannot extract requested metric from result struct.\n' ...
           'Available fields are:\n%s'], ...
           strjoin(fieldnames(R),', '));
end


function set_metric_axis_padding(ax,values,bottom_padding,top_padding)
%SET_METRIC_AXIS_PADDING Expand active yyaxis limits for visual separation.
%
% bottom_padding/top_padding are fractions of the CRLB data range.
% Increasing top_padding moves the CRLB curves downward without changing
% any numerical result.

    values = double(values(:));
    values = values(isfinite(values));

    if isempty(values)
        return;
    end

    ymin = min(values);
    ymax = max(values);
    yrange = ymax-ymin;

    if yrange <= 0
        yrange = max(abs(ymax),1);
    end

    lo = ymin-bottom_padding*yrange;
    hi = ymax+top_padding*yrange;

    ylim(ax,[lo hi]);
end


function c = paper_color(idx)
%PAPER_COLOR Use user's color(idx) if available.
%
% color(1), color(2)   : standard blue/red families
% color(11), color(12) : lightweight counterparts

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

    base = [ ...
        0.0000 0.4470 0.7410; ... % blue
        0.8500 0.3250 0.0980];    % red/orange

    if idx==1 || idx==2
        c = base(idx,:);
    elseif idx==11 || idx==12
        c0 = base(idx-10,:);
        lighten = 0.48;
        c = c0 + lighten*(1-c0);
    else
        error('Fallback defines color(1), color(2), color(11), and color(12).');
    end
end
