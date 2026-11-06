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
%   SSP-HBF with the same finite iteration budget N = 125
%   Deep-SSP-HBF with selected practical depth N = 125
%
% Fixed architecture:
%   G = 4
%
% Visual encoding:
%   Left y-axis  / Sum rate : blue family
%   Right y-axis / CRLB     : red/orange family
%   SSP-HBF reference       : darker color, solid line
%   SSP-HBF, N=125          : darker color, dotted line
%   Deep-SSP-HBF, N=125     : lightweight counterpart, dashed line
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
ssp125_file = '';

save_fig = false;
save_png = false;
save_eps = false;

% Final manuscript Pt range.
% The 30-dBm point is omitted because the 2000-iteration SSP-HBF reference
% is not sufficiently converged there.  Keeping Pt <= 25 dBm preserves the
% same 2000-iteration upper bound used throughout the paper.
max_Pt_dBm_plot = 25;

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
ssp125_name = 'data_Fig5_SSP125_generalization_Pt_Nt_G4_100ch.mat';

if isempty(base_file)
    base_file = locate_result_file(fig_dir,project_root,base_name);
end

if isempty(supp_file)
    supp_file = locate_result_file(fig_dir,project_root,supp_name);
end

if isempty(ssp125_file)
    ssp125_file = locate_result_file(fig_dir,project_root,ssp125_name);
end

fprintf('Base result:\n%s\n\n',base_file);
fprintf('Deep N=125 supplement:\n%s\n\n',supp_file);
fprintf('SSP N=125 supplement:\n%s\n\n',ssp125_file);

B = load(base_file);
D = load(supp_file);
E = load(ssp125_file);

%% -------------------- Sweep vectors --------------------
Pt_dBm_list = get_vector(B, ...
    {'Pt_dBm_list','Pt_list_dBm','Pt_list','Pt_dBm'}, ...
    [0 5 10 15 20 25 30]);

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

Pt_dBm_list_E = get_vector_optional(E, ...
    {'Pt_dBm_list','Pt_list_dBm','Pt_list','Pt_dBm'});

Nt_list_E = get_vector_optional(E, ...
    {'Nt_list','Ntx_list','Nt_values'});

if isempty(Pt_dBm_list_E) || isempty(Nt_list_E)
    error('SSP125 supplement is missing Pt/Nt sweep vectors.');
end

% Retain only the final manuscript Pt range.
keep_Pt = Pt_dBm_list <= max_Pt_dBm_plot;
Pt_dBm_list_full = Pt_dBm_list;
Pt_dBm_list = Pt_dBm_list(keep_Pt);

if ~isequal(Pt_dBm_list,Pt_dBm_list_E)
    error('Displayed Pt sweep and SSP125 supplement Pt sweep do not match.');
end

if ~isequal(Nt_list,Nt_list_E)
    error('Nt sweep and SSP125 supplement Nt sweep do not match.');
end

%% -------------------- Extract result structures --------------------
Pt_ref_struct = get_sweep_struct(B, ...
    {'Ptres','PtRes','Pt_result','PtResult','Pt_results'}, ...
    numel(Pt_dBm_list_full));

Nt_ref_struct = get_sweep_struct(B, ...
    {'Ntres','NtRes','Nt_result','NtResult','Nt_results'}, ...
    numel(Nt_list));

Pt_125_struct = get_sweep_struct(D, ...
    {'Pt125','Pt_125','Ptres125','PtRes125','Pt_result','PtResult','Pt_results'}, ...
    numel(Pt_dBm_list_full));

Nt_125_struct = get_sweep_struct(D, ...
    {'Nt125','Nt_125','Ntres125','NtRes125','Nt_result','NtResult','Nt_results'}, ...
    numel(Nt_list));

Pt_ssp125_struct = get_sweep_struct(E, ...
    {'PtSSP125','Pt_SSP125','Ptssp125','Pt_result','PtResult','Pt_results'}, ...
    numel(Pt_dBm_list));

Nt_ssp125_struct = get_sweep_struct(E, ...
    {'NtSSP125','Nt_SSP125','Ntssp125','Nt_result','NtResult','Nt_results'}, ...
    numel(Nt_list));

%% -------------------- Extract averaged physical metrics --------------------
% Long-run SSP-HBF reference from the base file.
SR_pt_ref = extract_metric_average(Pt_ref_struct, ...
    {'SRRef_avg','SR_ref_avg','SRSSP_avg','SR_SSP_avg','SRref_avg'}, ...
    {'SRRef','SR_ref','SRSSP','SR_SSP','SRref'}, ...
    numel(Pt_dBm_list_full));

CRLB_pt_ref = extract_metric_average(Pt_ref_struct, ...
    {'CRLBRef_avg','CRLB_ref_avg','CRLBSSP_avg','CRLB_SSP_avg','CRLBref_avg'}, ...
    {'CRLBRef','CRLB_ref','CRLBSSP','CRLB_SSP','CRLBref'}, ...
    numel(Pt_dBm_list_full));

SR_nt_ref = extract_metric_average(Nt_ref_struct, ...
    {'SRRef_avg','SR_ref_avg','SRSSP_avg','SR_SSP_avg','SRref_avg'}, ...
    {'SRRef','SR_ref','SRSSP','SR_SSP','SRref'}, ...
    numel(Nt_list));

CRLB_nt_ref = extract_metric_average(Nt_ref_struct, ...
    {'CRLBRef_avg','CRLB_ref_avg','CRLBSSP_avg','CRLB_SSP_avg','CRLBref_avg'}, ...
    {'CRLBRef','CRLB_ref','CRLBSSP','CRLB_SSP','CRLBref'}, ...
    numel(Nt_list));

% Conventional SSP-HBF with the same N=125 iteration budget.
SR_pt_ssp125 = extract_metric_average(Pt_ssp125_struct, ...
    {'SR_avg','SR125_avg','SR_125_avg'}, ...
    {'SR','SR125','SR_125'}, ...
    numel(Pt_dBm_list));

CRLB_pt_ssp125 = extract_metric_average(Pt_ssp125_struct, ...
    {'CRLB_avg','CRLB125_avg','CRLB_125_avg'}, ...
    {'CRLB','CRLB125','CRLB_125'}, ...
    numel(Pt_dBm_list));

SR_nt_ssp125 = extract_metric_average(Nt_ssp125_struct, ...
    {'SR_avg','SR125_avg','SR_125_avg'}, ...
    {'SR','SR125','SR_125'}, ...
    numel(Nt_list));

CRLB_nt_ssp125 = extract_metric_average(Nt_ssp125_struct, ...
    {'CRLB_avg','CRLB125_avg','CRLB_125_avg'}, ...
    {'CRLB','CRLB125','CRLB_125'}, ...
    numel(Nt_list));

% Deep-SSP-HBF with N=125 from the supplement.
SR_pt_125 = extract_metric_average(Pt_125_struct, ...
    {'SR_avg','SR125_avg','SR_125_avg','SRDeep_avg','SR_Deep_avg'}, ...
    {'SR','SR125','SR_125','SRDeep','SR_Deep'}, ...
    numel(Pt_dBm_list_full));

CRLB_pt_125 = extract_metric_average(Pt_125_struct, ...
    {'CRLB_avg','CRLB125_avg','CRLB_125_avg','CRLBDeep_avg','CRLB_Deep_avg'}, ...
    {'CRLB','CRLB125','CRLB_125','CRLBDeep','CRLB_Deep'}, ...
    numel(Pt_dBm_list_full));

% Apply the final manuscript Pt mask.
SR_pt_ref   = SR_pt_ref(keep_Pt);
CRLB_pt_ref = CRLB_pt_ref(keep_Pt);
SR_pt_125   = SR_pt_125(keep_Pt);
CRLB_pt_125 = CRLB_pt_125(keep_Pt);

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
Nssp = get_scalar(E,{'N_ssp','Nssp','N'},125);

if Nssp ~= Nexe
    error('Equal-iteration comparison requires N_ssp = Nexe.');
end

fprintf('Detected/used settings:\n');
fprintf('  G                 = %g\n',G);
fprintf('  Deep execution N  = %g\n',Nexe);
fprintf('  SSP iteration N   = %g\n',Nssp);
fprintf('  Training Pt       = %g dBm\n',Pt_train_dBm);
fprintf('  Training Nt       = %g\n\n',Nt_train);

%% ========================================================================
% Colors
%
% Standard dual-axis convention:
%   left / sum rate : blue, linear scale
%   right / CRLB    : red/orange
% For the Pt panel only, the CRLB axis uses a logarithmic scale because
% the low-SNR CRLB spans a much larger range than the moderate/high-SNR
% values. The Nt panel keeps the original linear CRLB scale.
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

line_ref = '-';
line_ssp125 = ':';
line_deep125 = '--';

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

hPt_sr_ref = plot(axPt, ...
    Pt_dBm_list,SR_pt_ref, ...
    line_ref, ...
    'Color',c_sr_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_sr, ...
    'MarkerSize',S.marker_size);

hPt_sr_ssp125 = plot(axPt, ...
    Pt_dBm_list,SR_pt_ssp125, ...
    line_ssp125, ...
    'Color',c_sr_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_sr, ...
    'MarkerSize',S.marker_size);

hPt_sr_deep = plot(axPt, ...
    Pt_dBm_list,SR_pt_125, ...
    line_deep125, ...
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

hPt_crlb_ref = plot(axPt, ...
    Pt_dBm_list,CRLB_pt_ref, ...
    line_ref, ...
    'Color',c_crlb_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_crlb, ...
    'MarkerSize',S.marker_size);

hPt_crlb_ssp125 = plot(axPt, ...
    Pt_dBm_list,CRLB_pt_ssp125, ...
    line_ssp125, ...
    'Color',c_crlb_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_crlb, ...
    'MarkerSize',S.marker_size);

hPt_crlb_deep = plot(axPt, ...
    Pt_dBm_list,CRLB_pt_125, ...
    line_deep125, ...
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

% The CRLB spans a large dynamic range over Pt. Use a logarithmic
% right y-axis so that the differences among the three methods remain
% visible at moderate/high SNR without removing the low-SNR points.
axPt.YAxis(2).Scale = 'log';

set_log_metric_axis_padding(axPt, ...
    [CRLB_pt_ref(:); CRLB_pt_ssp125(:); CRLB_pt_125(:)], ...
    crlb_pt_bottom_padding,crlb_pt_top_padding);

%% Common x-axis and styling
xlabel(axPt,'Transmit power (dBm)', ...
    'Interpreter','none');

xticks(axPt,Pt_dBm_list);
x_margin_Pt = 1.0;
xlim(axPt,[min(Pt_dBm_list)-x_margin_Pt, ...
           max(Pt_dBm_list)+x_margin_Pt]);

axPt.YAxis(1).Color = c_sr_ssp;
axPt.YAxis(2).Color = c_crlb_ssp;

% Shared compact legend for both Pt and Nt panels.
h_metric_sr_Pt = plot(axPt,nan,nan, ...
    'LineStyle','none','Color',c_sr_ssp, ...
    'Marker',marker_sr,'MarkerSize',S.marker_size);

h_metric_crlb_Pt = plot(axPt,nan,nan, ...
    'LineStyle','none','Color',c_crlb_ssp, ...
    'Marker',marker_crlb,'MarkerSize',S.marker_size);

h_ref_key_Pt = plot(axPt,nan,nan,line_ref, ...
    'Color',[0.25 0.25 0.25],'LineWidth',S.line_width);

h_ssp125_key_Pt = plot(axPt,nan,nan,line_ssp125, ...
    'Color',[0.25 0.25 0.25],'LineWidth',S.line_width);

h_deep125_key_Pt = plot(axPt,nan,nan,line_deep125, ...
    'Color',[0.60 0.60 0.60],'LineWidth',S.line_width);

h_train_Pt = plot(axPt,nan,nan, ...
    'x','Color','k','LineWidth',1.4,'MarkerSize',7.2);

legend_font_compact = max(S.legend_font-1.2,8.5);

lgdPt = legend(axPt, ...
    [h_metric_sr_Pt h_metric_crlb_Pt ...
     h_ref_key_Pt h_ssp125_key_Pt h_deep125_key_Pt h_train_Pt], ...
    {'Sum rate','CRLB', ...
     'SSP-HBF (ref.)', ...
     'SSP-HBF, T=125', ...
     'Deep-SSP-HBF, T=125', ...
     'Training setting'}, ...
    'Location','south', ...
    'FontName',S.font_name, ...
    'FontSize',legend_font_compact, ...
    'Box','off', ...
    'Interpreter','none');

try
    lgdPt.NumColumns = 2;
catch
end

try
    lgdPt.ItemTokenSize = [16 10];
catch
end

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

hNt_sr_ref = plot(axNt, ...
    Nt_list,SR_nt_ref, ...
    line_ref, ...
    'Color',c_sr_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_sr, ...
    'MarkerSize',S.marker_size);

hNt_sr_ssp125 = plot(axNt, ...
    Nt_list,SR_nt_ssp125, ...
    line_ssp125, ...
    'Color',c_sr_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_sr, ...
    'MarkerSize',S.marker_size);

hNt_sr_deep = plot(axNt, ...
    Nt_list,SR_nt_125, ...
    line_deep125, ...
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

hNt_crlb_ref = plot(axNt, ...
    Nt_list,CRLB_nt_ref, ...
    line_ref, ...
    'Color',c_crlb_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_crlb, ...
    'MarkerSize',S.marker_size);

hNt_crlb_ssp125 = plot(axNt, ...
    Nt_list,CRLB_nt_ssp125, ...
    line_ssp125, ...
    'Color',c_crlb_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_crlb, ...
    'MarkerSize',S.marker_size);

hNt_crlb_deep = plot(axNt, ...
    Nt_list,CRLB_nt_125, ...
    line_deep125, ...
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
    [CRLB_nt_ref(:); CRLB_nt_ssp125(:); CRLB_nt_125(:)], ...
    crlb_nt_bottom_padding,crlb_nt_top_padding);

%% Common x-axis and styling
xlabel(axNt,'Number of transmit antennas', ...
    'Interpreter','none');

xticks(axNt,Nt_list);
x_margin_Nt = 4;
xlim(axNt,[min(Nt_list)-x_margin_Nt, ...
           max(Nt_list)+x_margin_Nt]);

axNt.YAxis(1).Color = c_sr_ssp;
axNt.YAxis(2).Color = c_crlb_ssp;

% No legend in the Nt panel. The Pt-panel legend is shared by both
% subfigures because the visual encoding is identical.

plot_style('single',axNt);

%% ========================================================================
% Console summary
%% ========================================================================
fprintf('============================================================\n');
fprintf('Pt/Nt system-level generalization\n');
fprintf('============================================================\n');
fprintf('Connectivity           : G = %g\n',G);
fprintf('Deep execution depth   : %g\n',Nexe);
fprintf('Equal SSP budget       : %g iterations\n',Nssp);
fprintf('Training configuration : Pt = %g dBm, Nt = %g\n', ...
    Pt_train_dBm,Nt_train);
fprintf('Displayed Pt range     : %g to %g dBm\n', ...
    min(Pt_dBm_list),max(Pt_dBm_list));
fprintf('Pt-panel CRLB axis     : logarithmic\n');
fprintf('Note: 30 dBm omitted because the 2000-iteration SSP reference is under-converged.\n');

fprintf('\nTransmit-power sweep:\n');
fprintf(['Pt(dBm) | SR ref | SR SSP125 | SR Deep125 | ' ...
         'CRLB ref | CRLB SSP125 | CRLB Deep125\n']);

for i = 1:numel(Pt_dBm_list)
    fprintf('%7.1f | %7.4f | %9.4f | %10.4f | %8.4e | %11.4e | %12.4e\n', ...
        Pt_dBm_list(i), ...
        SR_pt_ref(i),SR_pt_ssp125(i),SR_pt_125(i), ...
        CRLB_pt_ref(i),CRLB_pt_ssp125(i),CRLB_pt_125(i));
end

fprintf('\nTransmit-antenna sweep:\n');
fprintf(['Nt | SR ref | SR SSP125 | SR Deep125 | ' ...
         'CRLB ref | CRLB SSP125 | CRLB Deep125\n']);

for i = 1:numel(Nt_list)
    fprintf('%3d | %7.4f | %9.4f | %10.4f | %8.4e | %11.4e | %12.4e\n', ...
        Nt_list(i), ...
        SR_nt_ref(i),SR_nt_ssp125(i),SR_nt_125(i), ...
        CRLB_nt_ref(i),CRLB_nt_ssp125(i),CRLB_nt_125(i));
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


function set_log_metric_axis_padding(ax,values,bottom_padding,top_padding)
%SET_LOG_METRIC_AXIS_PADDING Expand active log-scale yyaxis limits.
%
% bottom_padding/top_padding are fractions of the LOG10 data range.
% This preserves positivity and gives visually balanced padding when the
% metric spans multiple orders of magnitude.

    values = double(values(:));
    values = values(isfinite(values) & values>0);

    if isempty(values)
        return;
    end

    log_values = log10(values);
    lo = min(log_values);
    hi = max(log_values);
    log_range = hi-lo;

    if log_range <= 0
        log_range = 1;
    end

    lo = lo-bottom_padding*log_range;
    hi = hi+top_padding*log_range;

    ylim(ax,10.^[lo hi]);
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
