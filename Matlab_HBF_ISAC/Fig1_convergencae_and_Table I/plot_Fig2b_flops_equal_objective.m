clc
clear
close all

%% ========================================================================
% Fig. 4(b): FLOPs per channel at equal objective versus Nt (G = 4)
%
% Loads results/data_FLOPs_equal_performance_G4.mat produced by
% HBF_FLOPs_equal_performance_G4.m and plots it with the unified paper
% style plot_style.m.
%
% Fig. 4 is a two-panel figure, so the 'half' mode is used (as for the
% runtime/speedup pair). Colours follow Fig. 4(a) within the same figure:
%   SSP-HBF (direct implementation) -> S.baseline  (orange squares,
%                                      = "Direct implementation" in 4(a))
%   SSP-HBF                         -> S.method3   (blue triangles)
%   Deep-SSP-HBF                    -> S.proposed  (cyan circles)
% Set color_scheme = 'fig3' to use the Fig. 3 colours instead
% (SSP-HBF orange, Deep cyan, direct blue).
%% ========================================================================

%% -------------------- User options --------------------
G = 4;
style_mode    = 'half';  % print-size export (requires the 'half_print'
                               % case in plot_style.m), fonts print at ~6-7 pt
color_scheme  = 'fig4a';   % 'fig4a' or 'fig3'
Nt_max_plot   = 512;       % plot Nt up to this value
show_ratios   = true;      % annotate the ratios at the largest Nt
save_fig = false;
save_png = false;
save_eps = false;

manual_data_file = '';

%% -------------------- Paths and style --------------------
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

S = plot_style(style_mode);

switch color_scheme
    case 'fig4a'
        sty_dir  = S.baseline;  sty_ssp = S.method3;  sty_deep = S.proposed;
    case 'fig3'
        sty_dir  = S.method3;   sty_ssp = S.baseline; sty_deep = S.proposed;
    otherwise
        error('color_scheme must be ''fig4a'' or ''fig3''.');
end

%% -------------------- Load data --------------------
target_name = sprintf('data_FLOPs_equal_performance_G%d.mat',G);

if ~isempty(manual_data_file)
    data_file = manual_data_file;
else
    cand = {fullfile(fig_dir,'results',target_name), ...
            fullfile(fig_dir,target_name), ...
            fullfile(project_root,'results',target_name)};
    data_file = '';
    for k = 1:numel(cand)
        if exist(cand{k},'file'), data_file = cand{k}; break; end
    end
    if isempty(data_file)
        hit = dir(fullfile(project_root,'**',target_name));
        if isempty(hit)
            error('Cannot find %s under %s',target_name,project_root);
        end
        data_file = fullfile(hit(1).folder,hit(1).name);
    end
end

fprintf('Loading FLOPs data from:\n%s\n\n',data_file);
D = load(data_file);
Res = D.Res;

Nt   = [Res.Nt];
keep = Nt <= Nt_max_plot;
Nt   = Nt(keep);

fl_deep = [Res(keep).flops_deep];
fl_ssp  = [Res(keep).flops_ssp];
fl_dir  = [Res(keep).flops_dir];
eta     = [Res(keep).eta_deep];

%% -------------------- Figure --------------------
fig = figure( ...
    'Color',S.background, ...
    'Units','inches', ...
    'Position',[1 1 S.fig_size], ...
    'Name','FLOPs at equal objective', ...
    'NumberTitle','off');

ax = axes(fig);
hold(ax,'on');

h_dir = plot(ax,Nt,fl_dir,sty_dir.line, ...
    'Color',sty_dir.color, ...
    'LineWidth',S.line_width, ...
    'MarkerSize',S.marker_size);

h_ssp = plot(ax,Nt,fl_ssp,sty_ssp.line, ...
    'Color',sty_ssp.color, ...
    'LineWidth',S.line_width, ...
    'MarkerSize',S.marker_size);

h_deep = plot(ax,Nt,fl_deep,sty_deep.line, ...
    'Color',sty_deep.color, ...
    'LineWidth',S.line_width, ...
    'MarkerSize',S.marker_size);

set(ax,'XScale','log','YScale','log', ...
    'XTick',Nt, ...
    'XTickLabel',arrayfun(@num2str,Nt,'UniformOutput',false));
xlim(ax,[Nt(1)/1.12 Nt(end)*1.12]);

% Decade limits that enclose all curves.
ymin = 10^floor(log10(min([fl_deep fl_ssp fl_dir])));
ymax = 10^ceil(log10(max([fl_deep fl_ssp fl_dir])));
ylim(ax,[ymin ymax]);
set(ax,'YTick',10.^(log10(ymin):2:log10(ymax)));   % every other decade

xlabel(ax,'Number of transmit antennas','Interpreter','none');
ylabel(ax,'FLOPs per channel','Interpreter','none');


% Ratios at the largest plotted Nt, shown as vertical brackets that
% connect the two curves being compared:
%   upper bracket: SSP-HBF (direct) vs SSP-HBF  -> gain of the compact chain
%   lower bracket: SSP-HBF vs Deep-SSP-HBF      -> gain of deep unfolding
if show_ratios
    x_b  = Nt(end)*1.07;                 % bracket just right of the markers
    tick = 1.025;                        % half-width of the end ticks
    pad  = 1.18;                         % keep the brackets off the markers
    text_gap = 1.02;                     % label offset LEFT of the bracket
    pairs = {fl_dir(end), fl_ssp(end); fl_ssp(end), fl_deep(end)};
    for k = 1:2
        y_hi = pairs{k,1}/pad;  y_lo = pairs{k,2}*pad;
        plot(ax,[x_b x_b],[y_lo y_hi],'k-','LineWidth',0.9,'HandleVisibility','off');
        plot(ax,[x_b/tick x_b*tick],[y_hi y_hi],'k-','LineWidth',0.9,'HandleVisibility','off');
        plot(ax,[x_b/tick x_b*tick],[y_lo y_lo],'k-','LineWidth',0.9,'HandleVisibility','off');
        text(ax,x_b/text_gap,sqrt(pairs{k,1}*pairs{k,2}), ...
            sprintf('%.0fx',pairs{k,1}/pairs{k,2}), ...
            'HorizontalAlignment','right', ...
            'VerticalAlignment','middle', ...
            'FontName',S.font_name, ...
            'FontSize',S.annotation_font, ...
            'FontWeight','bold', ...
            'Interpreter','none');
    end
    xlim(ax,[Nt(1)/1.12 Nt(end)*1.12]);  % labels now sit left of the brackets
    fprintf('At Nt = %d: direct/SSP = %.1fx, SSP/Deep = %.1fx, direct/Deep = %.1fx\n', ...
        Nt(end),fl_dir(end)/fl_ssp(end),fl_ssp(end)/fl_deep(end),fl_dir(end)/fl_deep(end));
end

plot_style(style_mode,ax);

%% -------------------- Split legend --------------------
% MATLAB allows one legend per axes, so the second legend is placed on
% a second, invisible axes that exactly overlays the main axes.
%   legend 1 (upper left) : SSP-HBF (direct)
%   legend 2 (lower right): SSP-HBF, Deep-SSP-HBF
lg1 = legend(ax,h_dir, ...
    {'SSP-HBF (direct)'}, ...
    'Location','northwest', ...
    'FontName',S.font_name, ...
    'FontSize',S.legend_font, ...
    'Box',S.legend_box, ...
    'Interpreter','none');
lg1.ItemTokenSize = [12 6];          % short line samples (default [30 18])

ax2 = axes(fig,'Position',ax.Position,'Visible','off', ...
    'XLim',ax.XLim,'YLim',ax.YLim,'XScale','log','YScale','log');
hold(ax2,'on');
h_ssp2 = plot(ax2,nan,nan,sty_ssp.line, ...
    'Color',sty_ssp.color, ...
    'LineWidth',S.line_width, ...
    'MarkerSize',S.marker_size);
h_deep2 = plot(ax2,nan,nan,sty_deep.line, ...
    'Color',sty_deep.color, ...
    'LineWidth',S.line_width, ...
    'MarkerSize',S.marker_size);
lg2 = legend(ax2,[h_ssp2 h_deep2],{'SSP-HBF','Deep-SSP-HBF'}, ...
    'Location','southeast', ...      % use 'south' if it hits the 41x bracket
    'FontName',S.font_name, ...
    'FontSize',S.legend_font, ...
    'Box',S.legend_box, ...
    'Interpreter','none');
lg2.ItemTokenSize = [12 6];
drawnow;
ax2.Position = ax.Position;          % keep the overlay aligned with ax

%% -------------------- Console summary --------------------
fprintf('\n  Nt | Deep progress | GFLOPs Deep | GFLOPs SSP | GFLOPs direct | SSP/Deep | direct/SSP | direct/Deep\n');
for k = 1:numel(Nt)
    fprintf('%5d | %10.2f %% | %11.3f | %10.3f | %13.3f | %8.1f | %10.1f | %11.1f\n', ...
        Nt(k),100*eta(k),fl_deep(k)/1e9,fl_ssp(k)/1e9,fl_dir(k)/1e9, ...
        fl_ssp(k)/fl_deep(k),fl_dir(k)/fl_ssp(k),fl_dir(k)/fl_deep(k));
end

%% -------------------- Export --------------------
out_dir = fullfile(fig_dir,'results','figures_flops');
if ~exist(out_dir,'dir'), mkdir(out_dir); end

if save_fig
    savefig(fig,fullfile(out_dir,'Fig4b_flops_equal_objective.fig'));
end
if save_png
    print(fig,fullfile(out_dir,'Fig4b_flops_equal_objective.png'), ...
        '-dpng',sprintf('-r%d',S.export_dpi));
end
if save_eps
    print(fig,fullfile(out_dir,'Fig4b_flops_equal_objective.eps'), ...
        '-depsc2','-painters');
end
