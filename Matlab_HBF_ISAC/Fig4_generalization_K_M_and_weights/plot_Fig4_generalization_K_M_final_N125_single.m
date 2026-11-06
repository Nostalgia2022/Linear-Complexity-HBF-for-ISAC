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
%   Deep-SSP-HBF with selected practical depth N = 125
%
% Fixed architecture:
%   G = 4
%
% Plot semantics:
%   * left y-axis  : sum rate
%   * right y-axis : CRLB
%   * SSP-HBF      : standard/darker color, solid line
%   * Deep-SSP-HBF : lightweight counterpart color, dashed line
%   * black crosses mark the nominal training configuration
%
% This script uses plot_style('single') because each K/M panel is intended
% to occupy one full paper column, stacked vertically like the convergence figures.
%% ========================================================================

%% -------------------- User options --------------------
base_file = '';
supp_file = '';

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

%% -------------------- Validate data --------------------
required_base = {'K_list','M_list','Kres','Mres','K0','M0','G'};
required_supp = {'K_list','M_list','K125','M125','Nexe','G'};

for k = 1:numel(required_base)
    if ~isfield(B,required_base{k})
        error('Base result missing variable "%s".',required_base{k});
    end
end

for k = 1:numel(required_supp)
    if ~isfield(D,required_supp{k})
        error('Supplement result missing variable "%s".',required_supp{k});
    end
end

K_list = double(B.K_list(:).');
M_list = double(B.M_list(:).');

if ~isequal(K_list,double(D.K_list(:).')) || ...
   ~isequal(M_list,double(D.M_list(:).'))
    error('K_list/M_list mismatch between base and supplement.');
end

if double(B.G) ~= double(D.G)
    error('Connectivity G mismatch between base and supplement.');
end

Kres = ensure_averages(B.Kres);
Mres = ensure_averages(B.Mres);

K125 = ensure_averages(D.K125);
M125 = ensure_averages(D.M125);

K_train = double(B.K0);
M_train = double(B.M0);
G = double(B.G);
Nexe = double(D.Nexe);

%% -------------------- Required performance fields --------------------
check_fields(Kres,{'SRRef_avg','CRLBRef_avg'},'Kres');
check_fields(Mres,{'SRRef_avg','CRLBRef_avg'},'Mres');

check_fields(K125,{'SR_avg','CRLB_avg'},'K125');
check_fields(M125,{'SR_avg','CRLB_avg'},'M125');

%% ========================================================================
% Colors
%
% Use the user's paper palette when color.m exists:
%   SSP  : color(1)
%   Deep : color(11), lightweight counterpart
%% ========================================================================
c_ssp = paper_color(1);
c_deep = paper_color(11);

marker = 'o';

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

hK_ssp = plot(axK, ...
    K_list,Kres.SRRef_avg, ...
    '-', ...
    'Color',c_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker, ...
    'MarkerSize',S.marker_size);

hK_deep = plot(axK, ...
    K_list,K125.SR_avg, ...
    '--', ...
    'Color',c_deep, ...
    'LineWidth',S.line_width, ...
    'Marker',marker, ...
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

plot(axK, ...
    K_list,Kres.CRLBRef_avg, ...
    '-', ...
    'Color',c_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker, ...
    'MarkerSize',S.marker_size, ...
    'HandleVisibility','off');

plot(axK, ...
    K_list,K125.CRLB_avg, ...
    '--', ...
    'Color',c_deep, ...
    'LineWidth',S.line_width, ...
    'Marker',marker, ...
    'MarkerSize',S.marker_size, ...
    'HandleVisibility','off');

ylabel(axK,'CRLB', ...
    'Interpreter','none');

if ~isempty(idx_Ktrain)
    plot(axK,K_train,K125.CRLB_avg(idx_Ktrain), ...
        'x','Color','k','LineWidth',1.4,'MarkerSize',7.2, ...
        'HandleVisibility','off');
end

%% Common x-axis and styling
xlabel(axK,'Number of users', ...
    'Interpreter','none');

xticks(axK,K_list);
xlim(axK,[min(K_list) max(K_list)]);

% Both y-axes use neutral black labels/ticks.
axK.YAxis(1).Color = [0 0 0];
axK.YAxis(2).Color = [0 0 0];

% Compact semantic legend.
h_train_K = plot(axK,nan,nan, ...
    'x','Color','k','LineWidth',1.4,'MarkerSize',7.2);

legend(axK,[hK_ssp hK_deep h_train_K], ...
    {'SSP-HBF','Deep-SSP-HBF','Training setting'}, ...
    'Location','best', ...
    'FontName',S.font_name, ...
    'FontSize',S.legend_font, ...
    'Box','off', ...
    'Interpreter','none');

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

hM_ssp = plot(axM, ...
    M_list,Mres.SRRef_avg, ...
    '-', ...
    'Color',c_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker, ...
    'MarkerSize',S.marker_size);

hM_deep = plot(axM, ...
    M_list,M125.SR_avg, ...
    '--', ...
    'Color',c_deep, ...
    'LineWidth',S.line_width, ...
    'Marker',marker, ...
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

plot(axM, ...
    M_list,Mres.CRLBRef_avg, ...
    '-', ...
    'Color',c_ssp, ...
    'LineWidth',S.line_width, ...
    'Marker',marker, ...
    'MarkerSize',S.marker_size, ...
    'HandleVisibility','off');

plot(axM, ...
    M_list,M125.CRLB_avg, ...
    '--', ...
    'Color',c_deep, ...
    'LineWidth',S.line_width, ...
    'Marker',marker, ...
    'MarkerSize',S.marker_size, ...
    'HandleVisibility','off');

ylabel(axM,'CRLB', ...
    'Interpreter','none');

if ~isempty(idx_Mtrain)
    plot(axM,M_train,M125.CRLB_avg(idx_Mtrain), ...
        'x','Color','k','LineWidth',1.4,'MarkerSize',7.2, ...
        'HandleVisibility','off');
end

%% Common x-axis and styling
xlabel(axM,'Number of sensing targets', ...
    'Interpreter','none');

xticks(axM,M_list);
xlim(axM,[min(M_list) max(M_list)]);

axM.YAxis(1).Color = [0 0 0];
axM.YAxis(2).Color = [0 0 0];

h_train_M = plot(axM,nan,nan, ...
    'x','Color','k','LineWidth',1.4,'MarkerSize',7.2);

legend(axM,[hM_ssp hM_deep h_train_M], ...
    {'SSP-HBF','Deep-SSP-HBF','Training setting'}, ...
    'Location','best', ...
    'FontName',S.font_name, ...
    'FontSize',S.legend_font, ...
    'Box','off', ...
    'Interpreter','none');

plot_style('single',axM);

%% ========================================================================
% Console summary
%% ========================================================================
fprintf('============================================================\n');
fprintf('K/M system-level generalization\n');
fprintf('============================================================\n');
fprintf('Connectivity           : G = %d\n',G);
fprintf('Deep execution depth   : %d\n',Nexe);
fprintf('Training configuration : users = %d, targets = %d\n', ...
    K_train,M_train);

fprintf('\nUser sweep:\n');
fprintf('users | SR ref | SR Deep | CRLB ref | CRLB Deep\n');

for i = 1:numel(K_list)
    fprintf('%5d | %7.4f | %7.4f | %8.4e | %8.4e\n', ...
        K_list(i), ...
        Kres.SRRef_avg(i),K125.SR_avg(i), ...
        Kres.CRLBRef_avg(i),K125.CRLB_avg(i));
end

fprintf('\nTarget sweep:\n');
fprintf('targets | SR ref | SR Deep | CRLB ref | CRLB Deep\n');

for i = 1:numel(M_list)
    fprintf('%7d | %7.4f | %7.4f | %8.4e | %8.4e\n', ...
        M_list(i), ...
        Mres.SRRef_avg(i),M125.SR_avg(i), ...
        Mres.CRLBRef_avg(i),M125.CRLB_avg(i));
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

    base = [0.0000 0.4470 0.7410];

    if idx==1
        c = base;
    elseif idx==11
        lighten = 0.48;
        c = base + lighten*(1-base);
    else
        error('Fallback only defines color(1) and color(11).');
    end
end
