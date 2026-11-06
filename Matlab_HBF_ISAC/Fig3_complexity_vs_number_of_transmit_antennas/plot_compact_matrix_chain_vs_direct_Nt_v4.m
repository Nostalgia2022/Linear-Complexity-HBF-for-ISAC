clc
clear
close all

%% ========================================================================
% Compact matrix-chain implementation versus direct implementation
%
% Plotting only.
%
% This script uses the unified style file:
%   plot_style.m
%
% Layout:
%   'half' mode = two simple figures placed side by side in one LaTeX
%   two-column figure*.
%
% Outputs:
%   Fig. (a): measured per-iteration CPU time versus number of antennas
%   Fig. (b): speedup of compact implementation over direct implementation
%% ========================================================================

%% -------------------- User options --------------------
manual_data_file = '';  % set explicitly if auto-detection fails

save_fig = false;
save_png = false;
save_eps = false;

use_log_y_runtime = true;
use_log_x = true;   % recommended: antenna numbers double, so log-x avoids 32/64 overlap

%% -------------------- Paths --------------------
script_fullpath = mfilename('fullpath');

if isempty(script_fullpath)
    fig_dir = pwd;
else
    fig_dir = fileparts(script_fullpath);
end

project_root = fileparts(fig_dir);

% Make sure plot_style.m is visible.
if exist(fullfile(fig_dir,'plot_style.m'),'file')
    addpath(fig_dir);
elseif exist(fullfile(project_root,'plot_style.m'),'file')
    addpath(project_root);
elseif exist('plot_style','file') ~= 2
    error(['Cannot find plot_style.m. Put plot_style.m in this figure folder, ' ...
           'the project root, or on the MATLAB path.']);
end

% Unified style for two side-by-side panels.
S = plot_style('half');

%% -------------------- Locate result MAT file --------------------
if ~isempty(manual_data_file)

    if ~exist(manual_data_file,'file')
        error('manual_data_file does not exist:\n%s',manual_data_file);
    end

    data_file = manual_data_file;

else
    data_file = auto_find_complexity_file(fig_dir,project_root);
end

fprintf('Loading complexity data from:\n%s\n\n',data_file);

D = load(data_file);

%% -------------------- Read antenna numbers --------------------
[Nt_list,Nt_name] = get_first_numeric_vector(D,{ ...
    'Nt_list','N_t_list','Nt_values','Nt_vec','Nt_set','Nt_array','Ntx_list'});

if isempty(Nt_list)
    error('Could not identify the transmit-antenna vector in the MAT file.');
end

Nt_list = double(Nt_list(:).');

fprintf('Detected antenna vector: %s\n',Nt_name);

%% ========================================================================
% Detect benchmark components
%
% If several pairs are available:
%   runtime figure: prefer full iteration, otherwise ISAC core
%   speedup figure: show all detected components
%% ========================================================================

metrics = struct([]);

%% FIM core
[c,d,cn,dn] = get_metric_pair(D, ...
    {'time_fim_compact_avg','fim_compact_time_avg','runtime_fim_compact_avg', ...
     'T_fim_compact','t_fim_compact','time_compact_fim','compact_fim_time', ...
     'FIM_compact_avg','fim_time_compact'}, ...
    {'time_fim_direct_avg','fim_direct_time_avg','runtime_fim_direct_avg', ...
     'T_fim_direct','t_fim_direct','time_direct_fim','direct_fim_time', ...
     'FIM_direct_avg','fim_time_direct'});

if ~isempty(c)
    metrics(end+1).label = 'FIM core'; %#ok<SAGROW>
    metrics(end).compact = c;
    metrics(end).direct = d;
    metrics(end).compact_name = cn;
    metrics(end).direct_name = dn;
end

%% ISAC / gradient core
[c,d,cn,dn] = get_metric_pair(D, ...
    {'time_core_compact_avg','core_compact_time_avg','runtime_core_compact_avg', ...
     'time_isac_compact_avg','isac_compact_time_avg','runtime_isac_compact_avg', ...
     'T_core_compact','t_core_compact','time_compact_core','compact_core_time', ...
     'Core_compact_avg','ISAC_compact_avg','gradient_compact_time_avg', ...
     'time_gradient_compact_avg','compact_gradient_time_avg'}, ...
    {'time_core_direct_avg','core_direct_time_avg','runtime_core_direct_avg', ...
     'time_isac_direct_avg','isac_direct_time_avg','runtime_isac_direct_avg', ...
     'T_core_direct','t_core_direct','time_direct_core','direct_core_time', ...
     'Core_direct_avg','ISAC_direct_avg','gradient_direct_time_avg', ...
     'time_gradient_direct_avg','direct_gradient_time_avg'});

if ~isempty(c)
    metrics(end+1).label = 'ISAC core'; %#ok<SAGROW>
    metrics(end).compact = c;
    metrics(end).direct = d;
    metrics(end).compact_name = cn;
    metrics(end).direct_name = dn;
end

%% Full outer iteration
[c,d,cn,dn] = get_metric_pair(D, ...
    {'time_iter_compact_avg','iter_compact_time_avg','runtime_iter_compact_avg', ...
     'time_iteration_compact_avg','iteration_compact_time_avg', ...
     'T_iter_compact','t_iter_compact','time_compact_iter','compact_iter_time', ...
     'time_full_compact_avg','full_compact_time_avg','runtime_full_compact_avg', ...
     'Time_compact_avg','runtime_compact_avg','time_compact_avg'}, ...
    {'time_iter_direct_avg','iter_direct_time_avg','runtime_iter_direct_avg', ...
     'time_iteration_direct_avg','iteration_direct_time_avg', ...
     'T_iter_direct','t_iter_direct','time_direct_iter','direct_iter_time', ...
     'time_full_direct_avg','full_direct_time_avg','runtime_full_direct_avg', ...
     'Time_direct_avg','runtime_direct_avg','time_direct_avg'});

if ~isempty(c)
    metrics(end+1).label = 'Full iteration'; %#ok<SAGROW>
    metrics(end).compact = c;
    metrics(end).direct = d;
    metrics(end).compact_name = cn;
    metrics(end).direct_name = dn;
end

if isempty(metrics)
    fprintf('\nAvailable variables in the MAT file:\n');
    disp(fieldnames(D));
    error('No compact/direct timing pair was detected.');
end

%% -------------------- Validate and compute speedups --------------------
for m = 1:numel(metrics)

    metrics(m).compact = double(metrics(m).compact(:).');
    metrics(m).direct  = double(metrics(m).direct(:).');

    if numel(metrics(m).compact) ~= numel(Nt_list) || ...
       numel(metrics(m).direct)  ~= numel(Nt_list)

        error('Timing-vector length mismatch for "%s".',metrics(m).label);
    end

    metrics(m).speedup = ...
        metrics(m).direct ./ max(metrics(m).compact,eps);
end

%% -------------------- Select primary runtime metric --------------------
labels = {metrics.label};

idx_primary = find(strcmp(labels,'Full iteration'),1);

if isempty(idx_primary)
    idx_primary = find(strcmp(labels,'ISAC core'),1);
end

if isempty(idx_primary)
    idx_primary = 1;
end

P = metrics(idx_primary);

fprintf('\nPrimary timing curve: %s\n',P.label);
fprintf('  compact variable: %s\n',P.compact_name);
fprintf('  direct variable : %s\n\n',P.direct_name);

%% ========================================================================
% FIGURE 1: Per-iteration CPU time
%% ========================================================================
fig1 = figure( ...
    'Color',S.background, ...
    'Units','inches', ...
    'Position',[1 1 S.fig_size], ...
    'Name','Per-iteration CPU time', ...
    'NumberTitle','off');

ax1 = axes(fig1);
hold(ax1,'on');

h1 = plot(ax1,Nt_list,1e3*P.compact, ...
    S.proposed.line, ...
    'Color',S.proposed.color, ...
    'LineWidth',S.line_width, ...
    'MarkerSize',S.marker_size);

h2 = plot(ax1,Nt_list,1e3*P.direct, ...
    S.baseline.line, ...
    'Color',S.baseline.color, ...
    'LineWidth',S.line_width, ...
    'MarkerSize',S.marker_size);

if use_log_y_runtime
    set(ax1,'YScale','log');
end

xlabel(ax1,'Number of transmit antennas', ...
    'Interpreter','none');

ylabel(ax1,'CPU time (ms)', ...
    'Interpreter','none');

legend(ax1,[h1 h2], ...
    {'Compact matrix chain','Direct implementation'}, ...
    'Location','northwest', ...
    'FontName',S.font_name, ...
    'FontSize',S.legend_font, ...
    'Box',S.legend_box, ...
    'Interpreter','none');

if use_log_x
    set(ax1,'XScale','log');
    xlim(ax1,[Nt_list(1)/1.12, Nt_list(end)*1.12]);
else
    xlim(ax1,[min(Nt_list) max(Nt_list)]);
end

xticks(ax1,Nt_list);
xticklabels(ax1,string(Nt_list));

% Apply unified "half-panel" style last.
plot_style('half',ax1);

%% ========================================================================
% FIGURE 2: Speedup
%% ========================================================================
fig2 = figure( ...
    'Color',S.background, ...
    'Units','inches', ...
    'Position',[6 1 S.fig_size], ...
    'Name','Compact implementation speedup', ...
    'NumberTitle','off');

ax2 = axes(fig2);
hold(ax2,'on');

% We keep three distinct line styles if three benchmark scopes are present.
style_list = { ...
    S.proposed.line, ...
    S.method3.line, ...
    S.method4.line};

color_list = { ...
    S.proposed.color, ...
    S.method3.color, ...
    S.method4.color};

h = gobjects(numel(metrics),1);

for m = 1:numel(metrics)

    h(m) = plot(ax2,Nt_list,metrics(m).speedup, ...
        style_list{min(m,numel(style_list))}, ...
        'Color',color_list{min(m,numel(color_list))}, ...
        'LineWidth',S.line_width, ...
        'MarkerSize',S.marker_size);
end

yline(ax2,1,'k:', ...
    'LineWidth',1.0, ...
    'HandleVisibility','off');

xlabel(ax2,'Number of transmit antennas', ...
    'Interpreter','none');

ylabel(ax2,'Speedup (direct / compact)', ...
    'Interpreter','none');

legend(ax2,h,{metrics.label}, ...
    'Location','northwest', ...
    'FontName',S.font_name, ...
    'FontSize',S.legend_font, ...
    'Box',S.legend_box, ...
    'Interpreter','none');

if use_log_x
    set(ax2,'XScale','log');
    xlim(ax2,[Nt_list(1)/1.12, Nt_list(end)*1.12]);
else
    xlim(ax2,[min(Nt_list) max(Nt_list)]);
end

xticks(ax2,Nt_list);
xticklabels(ax2,string(Nt_list));

plot_style('half',ax2);

%% ========================================================================
% Console summary
%% ========================================================================
fprintf('============================================================\n');
fprintf('Compact matrix-chain benchmark\n');
fprintf('============================================================\n');

for m = 1:numel(metrics)

    fprintf('\n%s:\n',metrics(m).label);

    for i = 1:numel(Nt_list)

        fprintf(['  antennas=%4d | compact=%9.4f ms | direct=%9.4f ms | ' ...
                 'speedup=%7.2fx\n'], ...
            Nt_list(i), ...
            1e3*metrics(m).compact(i), ...
            1e3*metrics(m).direct(i), ...
            metrics(m).speedup(i));
    end

    fprintf('  Largest-array speedup: %.2fx\n',metrics(m).speedup(end));
end

fprintf('============================================================\n');

%% ========================================================================
% Optional export
%% ========================================================================
out_dir = fullfile(fig_dir,'results','figures_compact_vs_direct');

if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

if save_fig
    savefig(fig1,fullfile(out_dir,'Fig2a_complexity_per_iteration.fig'));
    savefig(fig2,fullfile(out_dir,'Fig2b_speedup_per_iteration.fig'));
end

if save_png
    print(fig1,fullfile(out_dir,'Fig2a_complexity_per_iteration.png'), ...
        '-dpng',sprintf('-r%d',S.export_dpi));

    print(fig2,fullfile(out_dir,'Fig2b_speedup_per_iteration.png'), ...
        '-dpng',sprintf('-r%d',S.export_dpi));
end

if save_eps
    print(fig1,fullfile(out_dir,'Fig2a_complexity_per_iteration.eps'), ...
        '-depsc2','-painters');

    print(fig2,fullfile(out_dir,'Fig2b_speedup_per_iteration.eps'), ...
        '-depsc2','-painters');
end

%% ========================================================================
% Local helpers
%% ========================================================================
function data_file = auto_find_complexity_file(fig_dir,project_root)

roots = { ...
    fullfile(fig_dir,'results'), ...
    fig_dir, ...
    fullfile(project_root,'Fig3_complexity_vs_number_of_transmit_antennas')};

candidates = {};

for rr = 1:numel(roots)

    if ~exist(roots{rr},'dir')
        continue;
    end

    hits = dir(fullfile(roots{rr},'*.mat'));

    for k = 1:numel(hits)
        candidates{end+1} = ...
            fullfile(hits(k).folder,hits(k).name); %#ok<AGROW>
    end
end

if isempty(candidates)
    error('No MAT files found in the expected complexity directories.');
end

candidates = unique(candidates,'stable');

best_score = -inf;
data_file = '';

for k = 1:numel(candidates)

    try
        info = whos('-file',candidates{k});
        names = lower(string({info.name}));

        score = 0;

        if any(contains(names,'nt'))
            score = score + 2;
        end

        if any(contains(names,'compact'))
            score = score + 3;
        end

        if any(contains(names,'direct'))
            score = score + 3;
        end

        if any(contains(names,'time')) || any(contains(names,'runtime'))
            score = score + 2;
        end

        if score > best_score
            best_score = score;
            data_file = candidates{k};
        end
    catch
    end
end

if isempty(data_file) || best_score < 8
    error(['Could not automatically identify the complexity MAT file. ' ...
           'Set manual_data_file at the top of this script.']);
end

end


function [x,name_found] = get_first_numeric_vector(S,candidates)

x = [];
name_found = '';

fields = fieldnames(S);

for c = 1:numel(candidates)

    idx = find(strcmpi(fields,candidates{c}),1);

    if isempty(idx)
        continue;
    end

    value = S.(fields{idx});

    if isnumeric(value) && isvector(value)
        x = value;
        name_found = fields{idx};
        return;
    end
end

end


function [compact,direct,compact_name,direct_name] = ...
    get_metric_pair(S,compact_candidates,direct_candidates)

[compact,compact_name] = ...
    get_first_numeric_vector(S,compact_candidates);

[direct,direct_name] = ...
    get_first_numeric_vector(S,direct_candidates);

if isempty(compact) || isempty(direct)
    compact = [];
    direct = [];
    compact_name = '';
    direct_name = '';
end

end
