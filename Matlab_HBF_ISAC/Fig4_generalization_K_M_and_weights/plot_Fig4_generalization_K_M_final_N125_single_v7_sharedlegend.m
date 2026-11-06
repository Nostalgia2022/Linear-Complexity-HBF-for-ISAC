clc
clear
close all

%% ========================================================================
% K/M generalization: final system-level performance plots
%
% Two separate MATLAB figures are generated for vertical stacking:
%   Fig. (a): performance versus number of users
%   Fig. (b): performance versus number of sensing targets
%
% Comparison:
%   SSP-HBF long-run reference
%   SSP-HBF with the same finite iteration budget N = 125
%   Deep-SSP-HBF with selected practical depth N = 125
%
% Fixed architecture:
%   G = 4
%
% Plot semantics:
%   * left y-axis  : sum rate
%   * right y-axis : CRLB
%   * SSP-HBF reference : standard/darker color, solid line
%   * SSP-HBF, N=125     : standard/darker color, dotted line
%   * Deep-SSP-HBF, N=125: lightweight counterpart, dashed line
%   * black crosses mark the nominal training configuration
%
% This script uses plot_style('single') because each K/M panel is intended
% to occupy one full paper column, stacked vertically like the convergence figures.
%% ========================================================================

%% -------------------- User options --------------------
base_file = '';
supp_file = '';
ssp125_file = '';

save_fig = false;
save_png = false;
save_eps = false;

% Visual separation for the dual-y-axis plots.
% This changes ONLY the displayed right-axis limits, not the CRLB data.
% A larger top padding moves the CRLB curves downward in the figure.
crlb_user_top_padding = 0.35;     % recommended 0.25--0.45
crlb_user_bottom_padding = 0.05;

% For the target-number figure we keep a milder padding.
crlb_target_top_padding = 0.15;
crlb_target_bottom_padding = 0.05;

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
base_name = 'data_Fig4_DeepSSP_generalization_K_M_G4_100ch.mat';
supp_name = 'data_Fig4_DeepSSP_generalization_K_M_G4_N125_supplement_100ch.mat';
ssp125_name = 'data_Fig4_SSP125_generalization_K_M_G4_100ch.mat';

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

%% -------------------- Validate data --------------------
required_base = {'K_list','M_list','Kres','Mres','K0','M0','G'};
required_supp = {'K_list','M_list','K125','M125','Nexe','G'};
required_ssp125 = {'K_list','M_list','KSSP125','MSSP125','N_ssp','G'};

for k = 1:numel(required_base)
    if ~isfield(B,required_base{k})
        error('Base result missing variable "%s".',required_base{k});
    end
end

for k = 1:numel(required_supp)
    if ~isfield(D,required_supp{k})
        error('Deep supplement missing variable "%s".',required_supp{k});
    end
end

for k = 1:numel(required_ssp125)
    if ~isfield(E,required_ssp125{k})
        error('SSP125 supplement missing variable "%s".',required_ssp125{k});
    end
end

K_list = double(B.K_list(:).');
M_list = double(B.M_list(:).');

if ~isequal(K_list,double(D.K_list(:).')) || ...
   ~isequal(M_list,double(D.M_list(:).')) || ...
   ~isequal(K_list,double(E.K_list(:).')) || ...
   ~isequal(M_list,double(E.M_list(:).'))
    error('K_list/M_list mismatch among the three result files.');
end

if double(B.G) ~= double(D.G) || double(B.G) ~= double(E.G)
    error('Connectivity G mismatch among result files.');
end

Kres = ensure_averages(B.Kres);
Mres = ensure_averages(B.Mres);

K125 = ensure_averages(D.K125);
M125 = ensure_averages(D.M125);

KSSP125 = ensure_averages(E.KSSP125);
MSSP125 = ensure_averages(E.MSSP125);

K_train = double(B.K0);
M_train = double(B.M0);
G = double(B.G);
Nexe = double(D.Nexe);
Nssp = double(E.N_ssp);

if Nssp ~= Nexe
    error('Equal-iteration comparison requires N_ssp = Nexe.');
end

%% -------------------- Required performance fields --------------------
check_fields(Kres,{'SRRef_avg','CRLBRef_avg'},'Kres');
check_fields(Mres,{'SRRef_avg','CRLBRef_avg'},'Mres');

check_fields(K125,{'SR_avg','CRLB_avg'},'K125');
check_fields(M125,{'SR_avg','CRLB_avg'},'M125');

check_fields(KSSP125,{'SR_avg','CRLB_avg'},'KSSP125');
check_fields(MSSP125,{'SR_avg','CRLB_avg'},'MSSP125');

%% ========================================================================
% Colors
%
% Metric is encoded by color following the usual dual-axis convention:
%   Left axis  / Sum rate : blue family
%   Right axis / CRLB     : red/orange family
%
% Algorithm is encoded by line style + shade:
%   SSP-HBF      : darker color, solid
%   Deep-SSP-HBF : lightweight counterpart, dashed
%
% With the user's color.m convention:
%   SR SSP     : color(1)
%   SR Deep    : color(11)
%   CRLB SSP   : color(2)
%   CRLB Deep  : color(12)
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
% FIGURE A: Generalization with respect to number of users
%% ========================================================================
figK = figure( ...
    'Color',S.background, ...
    'Units','inches', ...
    'Position',[1 1 S.fig_size], ...
    'Name','Generalization versus number of users', ...
    'NumberTitle','off');

axK = axes(figK);
hold(axK,'on');

%% Sum rate -- left axis
yyaxis(axK,'left');

hK_sr_ref = plot(axK, ...
    K_list,Kres.SRRef_avg, ...
    line_ref, ...
    'Color',c_sr_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_sr, ...
    'MarkerSize',S.marker_size);

hK_sr_ssp125 = plot(axK, ...
    K_list,KSSP125.SR_avg, ...
    line_ssp125, ...
    'Color',c_sr_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_sr, ...
    'MarkerSize',S.marker_size);

hK_sr_deep = plot(axK, ...
    K_list,K125.SR_avg, ...
    line_deep125, ...
    'Color',c_sr_deep, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_sr, ...
    'MarkerSize',S.marker_size);

ylabel(axK,'Sum rate (nat/s/Hz)', ...
    'Interpreter','none');

% Mark nominal training configuration on the Deep curve.
idx_Ktrain = find(K_list==K_train,1);

if ~isempty(idx_Ktrain)
    plot(axK,K_train,K125.SR_avg(idx_Ktrain), ...
        'x','Color','k','LineWidth',1.4,'MarkerSize',7.2, ...
        'HandleVisibility','off');
end

%% CRLB -- right axis
yyaxis(axK,'right');

hK_crlb_ref = plot(axK, ...
    K_list,Kres.CRLBRef_avg, ...
    line_ref, ...
    'Color',c_crlb_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_crlb, ...
    'MarkerSize',S.marker_size);

hK_crlb_ssp125 = plot(axK, ...
    K_list,KSSP125.CRLB_avg, ...
    line_ssp125, ...
    'Color',c_crlb_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_crlb, ...
    'MarkerSize',S.marker_size);

hK_crlb_deep = plot(axK, ...
    K_list,K125.CRLB_avg, ...
    line_deep125, ...
    'Color',c_crlb_deep, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_crlb, ...
    'MarkerSize',S.marker_size);

ylabel(axK,'CRLB', ...
    'Interpreter','none');

if ~isempty(idx_Ktrain)
    plot(axK,K_train,K125.CRLB_avg(idx_Ktrain), ...
        'x','Color','k','LineWidth',1.4,'MarkerSize',7.2, ...
        'HandleVisibility','off');
end

% Expand the CRLB right axis to visually separate it from the sum-rate
% curves.  The underlying CRLB values are unchanged.
set_metric_axis_padding(axK, ...
    [Kres.CRLBRef_avg(:); KSSP125.CRLB_avg(:); K125.CRLB_avg(:)], ...
    crlb_user_bottom_padding,crlb_user_top_padding);

%% Common x-axis and styling
xlabel(axK,'Number of users', ...
    'Interpreter','none');

xticks(axK,K_list);

% Leave a small horizontal margin so that the first and last markers are
% not attached to the plot boundaries.
x_margin_K = 0.2;
xlim(axK,[min(K_list)-x_margin_K, max(K_list)+x_margin_K]);

% Match each y-axis with its metric color.
axK.YAxis(1).Color = c_sr_ssp;
axK.YAxis(2).Color = c_crlb_ssp;

% Compact factorized legend:
% color/marker -> metric; line style -> algorithm/depth.
h_metric_sr_K = plot(axK,nan,nan, ...
    'LineStyle','none','Color',c_sr_ssp, ...
    'Marker',marker_sr,'MarkerSize',S.marker_size);

h_metric_crlb_K = plot(axK,nan,nan, ...
    'LineStyle','none','Color',c_crlb_ssp, ...
    'Marker',marker_crlb,'MarkerSize',S.marker_size);

h_ref_key_K = plot(axK,nan,nan,line_ref, ...
    'Color',[0.25 0.25 0.25],'LineWidth',S.line_width);

h_ssp125_key_K = plot(axK,nan,nan,line_ssp125, ...
    'Color',[0.25 0.25 0.25],'LineWidth',S.line_width);

h_deep125_key_K = plot(axK,nan,nan,line_deep125, ...
    'Color',[0.60 0.60 0.60],'LineWidth',S.line_width);

h_train_K = plot(axK,nan,nan, ...
    'x','Color','k','LineWidth',1.4,'MarkerSize',7.2);

% Shared legend for both K and M panels.  Since the visual encoding is
% identical in the two subfigures, we show the legend only in the first
% (user-number) panel to avoid repetition.
lgdK = legend(axK, ...
    [h_metric_sr_K h_metric_crlb_K ...
     h_ref_key_K h_ssp125_key_K h_deep125_key_K h_train_K], ...
    {'Sum rate','CRLB', ...
     'SSP-HBF reference', ...
     'SSP-HBF, N=125', ...
     'Deep-SSP-HBF, N=125', ...
     'Training setting'}, ...
    'Location','best', ...
    'FontName',S.font_name, ...
    'FontSize',S.legend_font, ...
    'Box','off', ...
    'Interpreter','none');

try
    lgdK.NumColumns = 2;
catch
end

plot_style('single',axK);

%% ========================================================================
% FIGURE B: Generalization with respect to number of sensing targets
%% ========================================================================
figM = figure( ...
    'Color',S.background, ...
    'Units','inches', ...
    'Position',[6.8 1 S.fig_size], ...
    'Name','Generalization versus number of sensing targets', ...
    'NumberTitle','off');

axM = axes(figM);
hold(axM,'on');

%% Sum rate -- left axis
yyaxis(axM,'left');

hM_sr_ref = plot(axM, ...
    M_list,Mres.SRRef_avg, ...
    line_ref, ...
    'Color',c_sr_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_sr, ...
    'MarkerSize',S.marker_size);

hM_sr_ssp125 = plot(axM, ...
    M_list,MSSP125.SR_avg, ...
    line_ssp125, ...
    'Color',c_sr_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_sr, ...
    'MarkerSize',S.marker_size);

hM_sr_deep = plot(axM, ...
    M_list,M125.SR_avg, ...
    line_deep125, ...
    'Color',c_sr_deep, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_sr, ...
    'MarkerSize',S.marker_size);

ylabel(axM,'Sum rate (nat/s/Hz)', ...
    'Interpreter','none');

idx_Mtrain = find(M_list==M_train,1);

if ~isempty(idx_Mtrain)
    plot(axM,M_train,M125.SR_avg(idx_Mtrain), ...
        'x','Color','k','LineWidth',1.4,'MarkerSize',7.2, ...
        'HandleVisibility','off');
end

%% CRLB -- right axis
yyaxis(axM,'right');

hM_crlb_ref = plot(axM, ...
    M_list,Mres.CRLBRef_avg, ...
    line_ref, ...
    'Color',c_crlb_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_crlb, ...
    'MarkerSize',S.marker_size);

hM_crlb_ssp125 = plot(axM, ...
    M_list,MSSP125.CRLB_avg, ...
    line_ssp125, ...
    'Color',c_crlb_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_crlb, ...
    'MarkerSize',S.marker_size);

hM_crlb_deep = plot(axM, ...
    M_list,M125.CRLB_avg, ...
    line_deep125, ...
    'Color',c_crlb_deep, ...
    'LineWidth',S.line_width, ...
    'Marker',marker_crlb, ...
    'MarkerSize',S.marker_size);

ylabel(axM,'CRLB', ...
    'Interpreter','none');

if ~isempty(idx_Mtrain)
    plot(axM,M_train,M125.CRLB_avg(idx_Mtrain), ...
        'x','Color','k','LineWidth',1.4,'MarkerSize',7.2, ...
        'HandleVisibility','off');
end

% Mild right-axis padding for visual clarity.
set_metric_axis_padding(axM, ...
    [Mres.CRLBRef_avg(:); MSSP125.CRLB_avg(:); M125.CRLB_avg(:)], ...
    crlb_target_bottom_padding,crlb_target_top_padding);

%% Common x-axis and styling
xlabel(axM,'Number of sensing targets', ...
    'Interpreter','none');

xticks(axM,M_list);

% Use the same small horizontal margin for the sensing-target sweep.
x_margin_M = 0.2;
xlim(axM,[min(M_list)-x_margin_M, max(M_list)+x_margin_M]);

axM.YAxis(1).Color = c_sr_ssp;
axM.YAxis(2).Color = c_crlb_ssp;

% No legend in the second panel: the first-panel legend is shared by both
% subfigures because the metric/algorithm encoding is identical.

plot_style('single',axM);

%% ========================================================================
% Console summary
%% ========================================================================
fprintf('============================================================\n');
fprintf('K/M system-level generalization\n');
fprintf('============================================================\n');
fprintf('Connectivity           : G = %d\n',G);
fprintf('Deep execution depth   : %d\n',Nexe);
fprintf('Equal SSP budget       : %d iterations\n',Nssp);
fprintf('Training configuration : users = %d, targets = %d\n', ...
    K_train,M_train);

fprintf('\nUser sweep:\n');
fprintf(['users | SR ref | SR SSP125 | SR Deep125 | ' ...
         'CRLB ref | CRLB SSP125 | CRLB Deep125\n']);

for i = 1:numel(K_list)
    fprintf('%5d | %7.4f | %9.4f | %10.4f | %8.4e | %11.4e | %12.4e\n', ...
        K_list(i), ...
        Kres.SRRef_avg(i),KSSP125.SR_avg(i),K125.SR_avg(i), ...
        Kres.CRLBRef_avg(i),KSSP125.CRLB_avg(i),K125.CRLB_avg(i));
end

fprintf('\nTarget sweep:\n');
fprintf(['targets | SR ref | SR SSP125 | SR Deep125 | ' ...
         'CRLB ref | CRLB SSP125 | CRLB Deep125\n']);

for i = 1:numel(M_list)
    fprintf('%7d | %7.4f | %9.4f | %10.4f | %8.4e | %11.4e | %12.4e\n', ...
        M_list(i), ...
        Mres.SRRef_avg(i),MSSP125.SR_avg(i),M125.SR_avg(i), ...
        Mres.CRLBRef_avg(i),MSSP125.CRLB_avg(i),M125.CRLB_avg(i));
end

fprintf('============================================================\n');

%% ========================================================================
% Optional export
%% ========================================================================
out_dir = fullfile(fig_dir,'results','figures_final_K_M');

if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

if save_fig
    savefig(figK,fullfile(out_dir,'Fig4a_generalization_users.fig'));
    savefig(figM,fullfile(out_dir,'Fig4b_generalization_targets.fig'));
end

if save_png
    print(figK,fullfile(out_dir,'Fig4a_generalization_users.png'), ...
        '-dpng',sprintf('-r%d',S.export_dpi));

    print(figM,fullfile(out_dir,'Fig4b_generalization_targets.png'), ...
        '-dpng',sprintf('-r%d',S.export_dpi));
end

if save_eps
    print(figK,fullfile(out_dir,'Fig4a_generalization_users.eps'), ...
        '-depsc2','-painters');

    print(figM,fullfile(out_dir,'Fig4b_generalization_targets.eps'), ...
        '-depsc2','-painters');
end

%% ========================================================================
% Local helpers
%% ========================================================================
function file = locate_result_file(fig_dir,project_root,name)

    candidates = { ...
        fullfile(fig_dir,'results',name), ...
        fullfile(fig_dir,name), ...
        fullfile(project_root,'Fig4_generalization_K_M','results',name), ...
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


function R = ensure_averages(R)

    fields = fieldnames(R);

    for kk = 1:numel(fields)

        name = fields{kk};

        if endsWith(name,'_avg') || endsWith(name,'_CI95')
            continue;
        end

        X = R.(name);

        if ~isnumeric(X) || ndims(X)~=2
            continue;
        end

        avg_name = [name '_avg'];

        if ~isfield(R,avg_name)
            R.(avg_name) = mean(X,2,'omitnan');
        end
    end
end


function check_fields(S,required,name)

    for k = 1:numel(required)
        if ~isfield(S,required{k})
            error('%s missing field "%s".',name,required{k});
        end
    end
end


function set_metric_axis_padding(ax,values,bottom_padding,top_padding)
%SET_METRIC_AXIS_PADDING Expand the active yyaxis limits for visual spacing.
%
% bottom_padding/top_padding are fractions of the data range.  Adding more
% top padding moves the plotted curve downward without altering the data.

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
% color(1)  : standard color
% color(11) : lightweight counterpart

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
