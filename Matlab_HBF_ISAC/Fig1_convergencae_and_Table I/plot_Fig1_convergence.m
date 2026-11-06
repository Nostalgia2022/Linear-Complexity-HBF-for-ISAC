clc
clear
close all

%% ========================================================================
% Convergence figure (Fig. 1a/1b) and Table I from ONE run:
%     data_TableI_v2_full_vs_52_N125_run200_100ch.mat
%
% Baseline  : SSP-HBF with the FULL decomposition (Algorithm 1,
%             I_AO = 200, I_in = 20), cold start.
% Reference : the same SSP-HBF after count_max iterations (per channel).
% Progress  : mean over channels of (f - f0)/(f_ref - f0)
%             (same definition as the previous figure; clipping to [0,1]
%              is applied only for display).
% Deep-SSP-HBF is shown through 200 layers; Table I is read at N_exe=125.
%
% Plotting rules (unchanged): plot_style('single'), solid algorithm
% curves, dashed lines only for thresholds/references, box off.
%% ========================================================================

%% -------------------- User options --------------------
threshold      = 0.99;
selected_depth = 125;          % N_exe
max_deep_depth = 200;
G_fig          = 4;            % architecture shown in the figure
show_ssp52     = false;        % also draw SSP-HBF (5/2) as a third curve
show_99_line   = true;         % draw the dashed 99% reference line
show_two_stage = true;         % two-stage FD->HBF baseline (horizontal level)
two_stage_file = '';           % '' = auto: data_TwoStage_FD_HBF_G<G>_nominal_100ch.mat

manual_data_file = '';

save_fig = false;
save_png = false;
save_eps = false;
write_table = true;

%% -------------------- Paths --------------------
script_fullpath = mfilename('fullpath');

if isempty(script_fullpath)
    fig_dir = pwd;
else
    fig_dir = fileparts(script_fullpath);
end

project_root = fileparts(fig_dir);

if exist(fullfile(fig_dir,'plot_style.m'),'file')
    addpath(fig_dir);
elseif exist(fullfile(project_root,'plot_style.m'),'file')
    addpath(project_root);
elseif exist('plot_style','file') ~= 2
    error(['Cannot find plot_style.m. Put it in this figure folder, ' ...
           'the project root, or on the MATLAB path.']);
end

S = plot_style('single');

%% -------------------- Locate data --------------------
target_name = 'data_TableI_v2_full_vs_52_N125_run200_100ch.mat';

if ~isempty(manual_data_file)
    data_file = manual_data_file;
else
    candidate_files = { ...
        fullfile(fig_dir,'results',target_name), ...
        fullfile(fig_dir,target_name), ...
        fullfile(project_root,'results',target_name)};
    data_file = '';
    for ii = 1:numel(candidate_files)
        if exist(candidate_files{ii},'file')
            data_file = candidate_files{ii};
            break;
        end
    end
    if isempty(data_file)
        hit = dir(fullfile(project_root,'**',target_name));
        if ~isempty(hit)
            data_file = fullfile(hit(1).folder,hit(1).name);
        end
    end
    if isempty(data_file)
        error('Cannot find %s under project root:\n%s',target_name,project_root);
    end
end

fprintf('Loading data from:\n%s\n\n',data_file);
D = load(data_file);
Res = D.Res;

%% -------------------- Normalized progress for all G --------------------
nG = numel(Res);
P = struct([]);

for iG = 1:nG
    r  = Res(iG);
    f0 = r.f0(:);
    fref = r.full.obj(:);                       % full SSP-HBF reference
    dd = max(fref - f0,1e-12);

    FA = fillmissing(r.full.obj_traj ,'previous',2);
    TA = fillmissing(r.full.time_traj,'previous',2);
    FB = fillmissing(r.d52.obj_traj  ,'previous',2);
    TB = fillmissing(r.d52.time_traj ,'previous',2);
    FD = r.deep.obj_traj;
    TD = r.deep.time_traj;

    q = struct();
    q.G = r.G;
    q.prog_ssp  = mean((FA - f0)./dd,1);   q.time_ssp  = mean(TA,1);
    q.prog_52   = mean((FB - f0)./dd,1);   q.time_52   = mean(TB,1);
    q.prog_deep = mean((FD - f0)./dd,1);   q.time_deep = mean(TD,1);

    % Table I quantities at N_exe
    q.eta_deep = q.prog_deep(selected_depth+1);
    q.eta_ssp  = q.prog_ssp(selected_depth+1);
    q.t_deep   = mean(TD(:,selected_depth+1));

    fD = FD(:,selected_depth+1);
    nC = numel(f0); it = nan(nC,1); tt = nan(nC,1); ok = false(nC,1);
    for ch = 1:nC
        k = find(FA(ch,:) >= fD(ch) - 1e-12,1,'first');
        if isempty(k)
            it(ch) = r.full.iter(ch); tt(ch) = r.full.time(ch);
        else
            it(ch) = k-1; tt(ch) = TA(ch,k); ok(ch) = true;
        end
    end
    q.reach_iter = mean(it);
    q.reach_time = mean(tt);
    q.reach_n    = nnz(ok);
    q.speedup    = q.reach_time/q.t_deep;
    q.n_deep_ge_ref = nnz(fD >= fref);

    k = find(q.prog_deep >= threshold,1);
    if isempty(k), q.N99_deep = []; else, q.N99_deep = k-1; end
    k = find(q.prog_ssp >= threshold,1);
    if isempty(k), q.N99_ssp = []; else, q.N99_ssp = k-1; end

    P = [P, q]; %#ok<AGROW>
end

%% -------------------- Select the architecture for the figure ----------
iF = find([P.G]==G_fig,1);
if isempty(iF), error('G=%d not in data file.',G_fig); end

progress_ssp  = min(max(P(iF).prog_ssp ,0),1);
progress_deep = min(max(P(iF).prog_deep,0),1);
progress_52   = min(max(P(iF).prog_52  ,0),1);
time_ssp  = P(iF).time_ssp;
time_deep = P(iF).time_deep;
time_52   = P(iF).time_52;

iteration_ssp  = 0:(numel(progress_ssp)-1);
iteration_deep = 0:(numel(progress_deep)-1);
iteration_52   = 0:(numel(progress_52)-1);

%% -------------------- Keep full Deep trajectory through 200 --------------------
max_deep_depth = min(max_deep_depth,iteration_deep(end));
keep_deep = iteration_deep <= max_deep_depth;

iteration_deep_plot = iteration_deep(keep_deep);
progress_deep_plot  = progress_deep(keep_deep);
time_deep_plot      = time_deep(keep_deep);

%% -------------------- Selected practical depth --------------------
idx_selected = find(iteration_deep_plot==selected_depth,1);
if isempty(idx_selected)
    error('Selected depth %d is not available.',selected_depth);
end
selected_progress = progress_deep_plot(idx_selected);
selected_time     = time_deep_plot(idx_selected);

%% -------------------- Equal-objective comparison points --------------------
% Deep-SSP-HBF after the selected depth versus SSP-HBF when it first
% attains the same objective on each channel (channel-averaged); these are
% exactly the Table I quantities, so figure and table agree.
% (Variable names *99 are kept from the previous version of this script.)
idx_deep99  = idx_selected;
deep_iter99 = selected_depth;
deep_time99 = P(iF).t_deep;

ssp_iter99  = round(P(iF).reach_iter);
idx_ssp99   = min(ssp_iter99+1,numel(progress_ssp));
ssp_time99  = P(iF).reach_time;
idx_ssp_t   = find(time_ssp >= ssp_time99,1,'first');
if isempty(idx_ssp_t), idx_ssp_t = numel(time_ssp); end

iteration_fraction = deep_iter99/ssp_iter99;

%% -------------------- Two-stage FD -> HBF baseline --------------------
% Non-iterative in the HBF domain (one decomposition of a stationary fully
% digital beamformer), so it is shown as a constant progress level. Its
% progress uses the SAME f0, reference, and channel average as the curves.
if show_two_stage
    if isempty(two_stage_file)
        ts_name = sprintf('data_TwoStage_FD_HBF_G%d_nominal_100ch.mat',G_fig);
        cand = {fullfile(fig_dir,'results',ts_name),fullfile(fig_dir,ts_name), ...
                fullfile(project_root,'results',ts_name)};
        for ii = 1:numel(cand)
            if exist(cand{ii},'file'), two_stage_file = cand{ii}; break; end
        end
        if isempty(two_stage_file)
            hit = dir(fullfile(project_root,'**',ts_name));
            if isempty(hit), error('Cannot find %s',ts_name); end
            two_stage_file = fullfile(hit(1).folder,hit(1).name);
        end
    end
    TSd = load(two_stage_file);
    rF  = Res([Res.G]==G_fig);
    if max(abs(TSd.f0(:) - rF.f0(:))) > 1e-8*max(abs(rF.f0(:)))
        warning('Two-stage f0 differs from the figure data: initialization mismatch.');
    end
    dd_F = max(rF.full.obj(:) - rF.f0(:),1e-12);
    prog_ts = mean((TSd.TS_obj(:) - rF.f0(:))./dd_F);   % mean of ratios
    time_ts = mean(TSd.TS_time(:));
    fprintf('Two-stage FD->HBF: progress %.4f, CPU time %.3f s/channel\n',prog_ts,time_ts);
    ts_color = S.method4.color;
end
runtime_fraction   = deep_time99/ssp_time99;

%% ========================================================================
% FIGURE 1: convergence versus iteration / unfolded layer
%% ========================================================================
fig1 = figure( ...
    'Color',S.background, ...
    'Units','inches', ...
    'Position',[1 1 S.fig_size], ...
    'Name','Convergence versus iteration', ...
    'NumberTitle','off');

ax1 = axes(fig1);
hold(ax1,'on');

% All method curves are SOLID under the global rule.
h_deep = plot(ax1, ...
    iteration_deep_plot,progress_deep_plot, ...
    S.proposed.line, ...
    'Color',S.proposed.color, ...
    'LineWidth',S.line_width, ...
    'MarkerSize',S.marker_size, ...
    'MarkerIndices',unique([1:15:numel(iteration_deep_plot), ...
                            numel(iteration_deep_plot)]));

h_ssp = plot(ax1, ...
    iteration_ssp,progress_ssp, ...
    S.baseline.line, ...
    'Color',S.baseline.color, ...
    'LineWidth',S.line_width, ...
    'MarkerSize',S.marker_size, ...
    'MarkerIndices',unique([1:120:numel(iteration_ssp), ...
                            numel(iteration_ssp)]));

if show_ssp52
    h_52 = plot(ax1,iteration_52,progress_52,'-^', ...
        'Color',[0.45 0.45 0.45], ...
        'LineWidth',S.line_width, ...
        'MarkerSize',S.marker_size, ...
        'MarkerIndices',unique([1:120:numel(iteration_52),numel(iteration_52)]));
end

% Threshold is a semantic reference, so dashed is appropriate.
if show_99_line
    yline(ax1,threshold,S.threshold_line, ...
        'Color',S.threshold_color, ...
        'LineWidth',1.7, ...
        'HandleVisibility','off');
end

% Filled markers: Deep at the selected depth, SSP when it reaches the same objective.
plot(ax1,deep_iter99,progress_deep_plot(idx_deep99), ...
    'ko','MarkerFaceColor','k','MarkerSize',6.3, ...
    'HandleVisibility','off');

plot(ax1,ssp_iter99,progress_ssp(idx_ssp99), ...
    'ks','MarkerFaceColor','k','MarkerSize',6.3, ...
    'HandleVisibility','off');

if show_two_stage
    h_ts = plot(ax1,[0 iteration_ssp(end)],[prog_ts prog_ts],'-.', ...
        'Color',ts_color,'LineWidth',S.line_width);

    % Exact progress value, placed right of the inset (inset spans
    % 40%-76% of the axes width) and just above the level line.
    ts_label_x   = 0.985*iteration_ssp(end);
    ts_label_gap = 0.012;                     % vertical offset (progress units)
    if abs(prog_ts - threshold) < 0.035
        ts_label_valign = 'top';  ts_label_y = prog_ts - ts_label_gap;   % avoid the 99% line
    else
        ts_label_valign = 'bottom'; ts_label_y = prog_ts + ts_label_gap;
    end
    text(ax1,ts_label_x,ts_label_y, ...
        sprintf('%.1f%%',100*prog_ts), ...
        'HorizontalAlignment','right', ...
        'VerticalAlignment',ts_label_valign, ...
        'Color',ts_color, ...
        'FontName',S.font_name, ...
        'FontSize',S.annotation_font, ...
        'FontWeight','bold', ...
        'BackgroundColor','w', ...
        'Margin',0.5, ...
        'Interpreter','none');
end

xlabel(ax1,'Iteration / unfolded layer', ...
    'Interpreter','none');

ylabel(ax1,'Normalized objective progress', ...
    'Interpreter','none');

if show_ssp52
    leg_h1 = [h_deep h_ssp h_52]; leg_t = {'Deep-SSP-HBF','SSP-HBF','SSP-HBF (5/2)'};
else
    leg_h1 = [h_deep h_ssp];      leg_t = {'Deep-SSP-HBF','SSP-HBF'};
end
if show_two_stage
    leg_h1 = [leg_h1 h_ts]; leg_t = [leg_t {'Two-stage FD-HBF'}];
end
legend(ax1,leg_h1, ...
    leg_t, ...
    'Location','southeast', ...
    'FontName',S.font_name, ...
    'FontSize',S.legend_font, ...
    'Box',S.legend_box, ...
    'Interpreter','none');

xlim(ax1,[0 iteration_ssp(end)]);
y_low = 0;
if show_two_stage && prog_ts < 0.02
    y_low = prog_ts - 0.05;   % keep the baseline level visible
end
ylim(ax1,[y_low 1.015]);

% Plain-text annotation only.
if show_99_line
    text(ax1,0.45*iteration_ssp(end),0.963, ...
        '99% reference improvement', ...
        'FontName',S.font_name, ...
        'FontSize',S.annotation_font, ...
        'FontWeight','bold', ...
        'Interpreter','none');
end

plot_style('single',ax1);

%% -------------------- Inset: complete Deep trajectory through 200 --------------------
drawnow;
p = ax1.Position;

ax_in = axes(fig1, ...
    'Position',[ ...
        p(1)+0.40*p(3), ...
        p(2)+0.27*p(4), ...
        0.36*p(3), ...
        0.35*p(4)]);

hold(ax_in,'on');

plot(ax_in, ...
    iteration_deep_plot,progress_deep_plot, ...
    S.proposed.line, ...
    'Color',S.proposed.color, ...
    'LineWidth',S.inset_line_width, ...
    'MarkerSize',S.inset_marker_size, ...
    'MarkerIndices',unique([1:15:numel(iteration_deep_plot), ...
                            numel(iteration_deep_plot)]));

if show_99_line
    yline(ax_in,threshold,S.threshold_line, ...
        'Color',S.threshold_color, ...
        'LineWidth',1.15);
end

% Selected practical depth.
plot(ax_in,selected_depth,selected_progress, ...
    'kd','MarkerFaceColor','k','MarkerSize',5.5);

% Reference guide is allowed to use a non-solid line.
xline(ax_in,selected_depth,S.reference_line, ...
    'Color',S.reference_color, ...
    'LineWidth',0.95);

% Avoid mathematical notation in annotations.
text(ax_in,selected_depth+5,0.950, ...
    sprintf('Selected depth = %d',selected_depth), ...
    'HorizontalAlignment','left', ...
    'FontName',S.font_name, ...
    'FontSize',S.inset_axis_font, ...
    'Interpreter','none');

xlim(ax_in,[0 210]);
ylim(ax_in,[0.88 1.005]);

set(ax_in, ...
    'FontName',S.font_name, ...
    'FontSize',S.inset_axis_font, ...
    'LineWidth',0.75, ...
    'Box','off', ...
    'Layer','top', ...
    'TickLabelInterpreter','none');

grid(ax_in,'on');

if isprop(ax_in,'GridAlpha')
    ax_in.GridAlpha = S.grid_alpha;
end

%% ========================================================================
% FIGURE 2: convergence versus serial CPU time
%% ========================================================================
fig2 = figure( ...
    'Color',S.background, ...
    'Units','inches', ...
    'Position',[6.5 1 S.fig_size], ...
    'Name','Convergence versus CPU time', ...
    'NumberTitle','off');

ax2 = axes(fig2);
hold(ax2,'on');

% Zero time cannot appear on a logarithmic axis.
valid_deep = isfinite(time_deep_plot) & time_deep_plot>0;
valid_ssp = isfinite(time_ssp) & time_ssp>0;

td = time_deep_plot(valid_deep);
pd = progress_deep_plot(valid_deep);

ts = time_ssp(valid_ssp);
ps = progress_ssp(valid_ssp);

h_deep_time = plot(ax2, ...
    td,pd, ...
    S.proposed.line, ...
    'Color',S.proposed.color, ...
    'LineWidth',S.line_width, ...
    'MarkerSize',S.marker_size, ...
    'MarkerIndices',unique([1:15:numel(td),numel(td)]));

h_ssp_time = plot(ax2, ...
    ts,ps, ...
    S.baseline.line, ...
    'Color',S.baseline.color, ...
    'LineWidth',S.line_width, ...
    'MarkerSize',S.marker_size, ...
    'MarkerIndices',unique([1:120:numel(ts),numel(ts)]));

if show_ssp52
    v52 = isfinite(time_52) & time_52>0;
    h_52_time = plot(ax2,time_52(v52),progress_52(v52),'-^', ...
        'Color',[0.45 0.45 0.45], ...
        'LineWidth',S.line_width, ...
        'MarkerSize',S.marker_size, ...
        'MarkerIndices',unique([1:120:nnz(v52),nnz(v52)]));
end

set(ax2,'XScale','log');

if show_two_stage
    h_ts_time = plot(ax2,time_ts,prog_ts,'p', ...
        'Color',ts_color,'MarkerFaceColor',ts_color,'MarkerSize',S.marker_size+3);

    % Total CPU time (stage 1 + stage 2), next to the star. Placed below the
    % level line if the star is close to the 99% line, otherwise above it.
    if abs(prog_ts - threshold) < 0.035
        ts_t_valign = 'top';    ts_t_y = prog_ts - 0.015;
    else
        ts_t_valign = 'bottom'; ts_t_y = prog_ts + 0.015;
    end
    text(ax2,time_ts,ts_t_y, ...
        sprintf('%.3f s',time_ts), ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment',ts_t_valign, ...
        'Color',ts_color, ...
        'FontName',S.font_name, ...
        'FontSize',S.annotation_font-0.5, ...
        'FontWeight','bold', ...
        'BackgroundColor','w', ...
        'Margin',0.5, ...
        'Interpreter','none');
end

if show_99_line
    yline(ax2,threshold,S.threshold_line, ...
        'Color',S.threshold_color, ...
        'LineWidth',1.7, ...
        'HandleVisibility','off');
end

plot(ax2,deep_time99,selected_progress, ...
    'ko','MarkerFaceColor','k','MarkerSize',6.3, ...
    'HandleVisibility','off');

plot(ax2,ssp_time99,progress_ssp(idx_ssp_t), ...
    'ks','MarkerFaceColor','k','MarkerSize',6.3, ...
    'HandleVisibility','off');

% Explicitly mark the equal-accuracy runtime difference at the 99% level.
% Draw two SOLID black vertical lines downward from the two 99% points.
% Runtime labels are placed near the lower ends, while the speedup is moved
% further down to keep the threshold region visually clean.
speedup_99 = ssp_time99/deep_time99;

y_top_deep = selected_progress;
y_top_ssp  = progress_ssp(idx_ssp_t);
y_line_bottom = 0.74;
y_time_label = 0.725;
y_speedup = 0.67;

plot(ax2,[deep_time99 deep_time99],[y_line_bottom y_top_deep], ...
    'k-','LineWidth',1.20,'HandleVisibility','off');

plot(ax2,[ssp_time99 ssp_time99],[y_line_bottom y_top_ssp], ...
    'k-','LineWidth',1.20,'HandleVisibility','off');

% On a logarithmic x-axis the geometric mean is the visual midpoint.
x_mid_compare = sqrt(deep_time99*ssp_time99);

% Runtime labels below the two vertical lines.
text(ax2,deep_time99,y_time_label, ...
    sprintf('%.3f s',deep_time99), ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','top', ...
    'FontName',S.font_name, ...
    'FontSize',S.annotation_font-0.5, ...
    'BackgroundColor','w', ...
    'Margin',1, ...
    'Interpreter','none');

text(ax2,ssp_time99,y_time_label, ...
    sprintf('%.3f s',ssp_time99), ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','top', ...
    'FontName',S.font_name, ...
    'FontSize',S.annotation_font-0.5, ...
    'BackgroundColor','w', ...
    'Margin',1, ...
    'Interpreter','none');

% Equal-accuracy speedup, intentionally separated from the threshold line.
text(ax2,x_mid_compare,y_speedup, ...
    sprintf('%.1fx faster at equal objective',speedup_99), ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','top', ...
    'FontName',S.font_name, ...
    'FontSize',S.annotation_font, ...
    'FontWeight','bold', ...
    'BackgroundColor','w', ...
    'Margin',1, ...
    'Interpreter','none');

% Selected depth is shown as a filled diamond.
if selected_time>0
    plot(ax2,selected_time,selected_progress, ...
        'kd','MarkerFaceColor','k','MarkerSize',6.0, ...
        'HandleVisibility','off');
end

xlabel(ax2,'Serial CPU time (s/channel)', ...
    'Interpreter','none');

ylabel(ax2,'Normalized objective progress', ...
    'Interpreter','none');

if show_ssp52
    leg_h2 = [h_deep_time h_ssp_time h_52_time];
else
    leg_h2 = [h_deep_time h_ssp_time];
end
if show_two_stage
    leg_h2 = [leg_h2 h_ts_time];
end
legend(ax2,leg_h2, ...
    leg_t, ...
    'Location','southeast', ...
    'FontName',S.font_name, ...
    'FontSize',S.legend_font, ...
    'Box',S.legend_box, ...
    'Interpreter','none');

ylim(ax2,[y_low 1.015]);

xmin = 0.85*min([td ts]);
xmax = 1.10*max([td ts]);
if show_two_stage
    xmin = min(xmin,0.85*time_ts); xmax = max(xmax,1.10*time_ts);
    plot(ax2,[xmin xmax],[prog_ts prog_ts],'-.','Color',ts_color, ...
        'LineWidth',S.line_width,'HandleVisibility','off');
end
xlim(ax2,[xmin xmax]);

if show_99_line
    x_text = 10^(log10(xmin) + 0.50*(log10(xmax)-log10(xmin)));
    text(ax2,x_text,0.963, ...
        '99% reference improvement', ...
        'FontName',S.font_name, ...
        'FontSize',S.annotation_font, ...
        'FontWeight','bold', ...
        'Interpreter','none');
end

plot_style('single',ax2);

%% ========================================================================
% Console summary
%% ========================================================================
fprintf('\n============================================================\n');
fprintf('COMMON long-run SSP reference\n');
fprintf('Deep-SSP-HBF plotted through depth %d\n',max_deep_depth);
fprintf('Selected practical depth = %d\n',selected_depth);
fprintf('============================================================\n');

fprintf('Deep progress at selected depth = %.5f (%.2f%%) | at %d: %.2f%%\n', ...
    P(iF).prog_deep(selected_depth+1),100*P(iF).prog_deep(selected_depth+1), ...
    max_deep_depth,100*P(iF).prog_deep(max_deep_depth+1));

fprintf('\nEqual-objective comparison (Deep at the selected depth):\n');
fprintf('Deep-SSP-HBF : %d layers, %.5f s/channel\n', ...
    deep_iter99,deep_time99);
fprintf('SSP-HBF      : %d iterations, %.5f s/channel\n', ...
    ssp_iter99,ssp_time99);

fprintf('\nDeep uses %.2f%% of SSP iterations.\n', ...
    100*iteration_fraction);
fprintf('Iteration reduction = %.2f%%.\n', ...
    100*(1-iteration_fraction));

fprintf('Deep uses %.2f%% of SSP runtime.\n', ...
    100*runtime_fraction);
fprintf('Runtime reduction = %.2f%%.\n', ...
    100*(1-runtime_fraction));
fprintf('Equal-objective speedup = %.2fx.\n', ...
    1/runtime_fraction);

fprintf('============================================================\n');

%% ========================================================================
% Optional export
%% ========================================================================
out_dir = fullfile(fig_dir,'results','figures_fullSSP_ref');

if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

if save_fig
    savefig(fig1,fullfile(out_dir,'Fig1a_convergence_vs_iteration.fig'));
    savefig(fig2,fullfile(out_dir,'Fig1b_convergence_vs_cputime.fig'));
end

if save_png
    print(fig1,fullfile(out_dir,'Fig1a_convergence_vs_iteration.png'), ...
        '-dpng',sprintf('-r%d',S.export_dpi));

    print(fig2,fullfile(out_dir,'Fig1b_convergence_vs_cputime.png'), ...
        '-dpng',sprintf('-r%d',S.export_dpi));
end

if save_eps
    print(fig1,fullfile(out_dir,'Fig1a_convergence_vs_iteration.eps'), ...
        '-depsc2','-painters');

    print(fig2,fullfile(out_dir,'Fig1b_convergence_vs_cputime.eps'), ...
        '-depsc2','-painters');
end

%% ========================================================================
% Table I (same run, same reference, same normalization as the figure)
%% ========================================================================
fprintf('\n============================================================\n');
fprintf('Table I data | ref = SSP-HBF (full) after %d iterations\n',numel(P(1).prog_ssp)-1);
fprintf('============================================================\n');
fprintf('  G | SSP eta@%d | to reach Deep: iter, time (reached) | Deep eta@%d | Deep time | speedup | Deep>=ref | N99 Deep / SSP\n', ...
    selected_depth,selected_depth);
for q = P
    fprintf('  %d | %8.2f %% | %6.0f, %7.3f s (%3d/100)        | %8.2f %% | %7.3f s | %5.1fx | %3d/100 | %s / %s\n', ...
        q.G,100*q.eta_ssp,q.reach_iter,q.reach_time,q.reach_n, ...
        100*q.eta_deep,q.t_deep,q.speedup,q.n_deep_ge_ref, ...
        num2str(q.N99_deep),num2str(q.N99_ssp));
end

if write_table
    L = {};
    L{end+1} = '\begin{table}[t]';
    L{end+1} = '\centering';
    L{end+1} = ['\caption{Achieved fraction $\eta$ of the reference objective ' ...
        'improvement and per-channel CPU time at the nominal operating point, with ' ...
        'both methods started from the same initialization. The reference is SSP-HBF ' ...
        'after $2000$ iterations; ``to reach Deep'''' denotes the cost of SSP-HBF to ' ...
        'attain the objective of Deep-SSP-HBF on each channel.}'];
    L{end+1} = '\label{tab:runtime}';
    L{end+1} = '\setlength{\tabcolsep}{4pt}';
    L{end+1} = '\renewcommand{\arraystretch}{1.08}';
    L{end+1} = '\begin{tabular}{c ccc cc c}';
    L{end+1} = '\hline';
    L{end+1} = '& \multicolumn{3}{c}{SSP-HBF} & \multicolumn{2}{c}{Deep-SSP-HBF} & \\';
    L{end+1} = '\cline{2-4}\cline{5-6}';
    L{end+1} = '& $\eta$ (\%) & \multicolumn{2}{c}{To reach Deep} & $\eta$ (\%) & Time & \\';
    L{end+1} = sprintf('$G$ & $%d$ iter. & Iter. & Time (s) & $N_{\\rm exe}{=}%d$ & (s) & Speedup \\\\', ...
        selected_depth,selected_depth);
    L{end+1} = '\hline';
    for q = P
        L{end+1} = sprintf('%d & %.1f & %.0f & %.3f & %.1f & %.3f & $%.1f\\times$ \\\\', ...
            q.G,100*q.eta_ssp,q.reach_iter,q.reach_time,100*q.eta_deep,q.t_deep,q.speedup); %#ok<SAGROW>
    end
    L{end+1} = '\hline';
    L{end+1} = '\end{tabular}';
    L{end+1} = '\end{table}';
    fprintf('\n'); fprintf('%s\n',L{:});

    tab_dir = fullfile(fig_dir,'results');
    if ~exist(tab_dir,'dir'), mkdir(tab_dir); end
    fid = fopen(fullfile(tab_dir,'TableI_fullSSP_ref.tex'),'w');
    fprintf(fid,'%s\n',L{:}); fclose(fid);
end
