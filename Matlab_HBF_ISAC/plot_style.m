function S = plot_style(mode, ax)
%PLOT_STYLE Unified plotting style for the SSP-HBF / Deep-SSP-HBF paper.
%
% Usage
% -----
%   S = plot_style('single');
%   S = plot_style('single', gca);
%
%   S = plot_style('half');
%   S = plot_style('half', gca);
%
% Modes
% -----
% 'single'
%   One MATLAB figure is used as one large paper panel.
%   Use this for important/detail-heavy figures, e.g., convergence curves.
%
% 'half'
%   Two MATLAB figures are placed side by side in one two-column LaTeX
%   figure*. Use this for simple paired figures, e.g., runtime and speedup.
%
% General paper-style rules
% -------------------------
% 1) Times New Roman throughout.
% 2) No figure title.
% 3) Axes box OFF.
% 4) Plain-text axis labels. Avoid symbols such as N_t, P_t, etc.
% 5) White background.
% 6) DEFAULT: distinguish methods by COLOR + MARKER only.
%    All method curves use SOLID lines unless a particular figure has a
%    specific reason to use another line style.
% 7) Reference/threshold lines may still use dashed/dotted styles.
% 8) Tight outer margins for compact EPS/PDF export.
%
% Tianyu Fang, 2026

if nargin < 1 || isempty(mode)
    mode = 'single';
end

mode = lower(string(mode));

%% Shared visual language
S.font_name = 'Times New Roman';

% Default method styles: SOLID lines, distinguished by color + marker.
S.proposed.color = [0.0000 0.7000 0.7000];
S.proposed.line  = '-o';

S.baseline.color = [0.8500 0.3250 0.0980];
S.baseline.line  = '-s';

S.method3.color  = [0.0000 0.4470 0.7410];
S.method3.line   = '-^';

S.method4.color  = [0.4940 0.1840 0.5560];
S.method4.line   = '-d';

% Non-method visual aids.
S.threshold_color = [0.30 0.30 0.30];
S.threshold_line = '--';

S.reference_color = [0.45 0.45 0.45];
S.reference_line = ':';

S.box = 'off';
S.grid = 'on';
S.minor_grid = 'off';
S.legend_box = 'on';
S.interpreter = 'none';

S.background = 'w';
S.tight_margin = 0.015;
S.export_dpi = 300;

%% Two publication cases
switch mode

    case {"single","large","one-panel"}
        S.mode = 'single';
        S.fig_size = [5.30 4.10];

        S.axis_font = 6.5;          % -> about 5.6 pt printed (Fig. 3: about 5.2 pt)
        S.label_font = 7.0;         % -> about 6.0 pt (Fig. 3: about 5.6 pt)
        S.legend_font = 6.0;        % -> about 5.2 pt (Fig. 3: about 4.9 pt)
        S.annotation_font = 6.0;

        S.line_width = 1.70;
        S.marker_size = 5.5;
        S.marker_edge_width = 1.1;

        S.axes_line_width = 0.90;
        S.grid_alpha = 0.16;

        S.inset_axis_font = 8.5;
        S.inset_line_width = 1.30;
        S.inset_marker_size = 4.1;

    case {"half_print"}
        S.mode = 'half';
        S.fig_size = [2.00 1.60];

        S.axis_font = 8.0;
        S.label_font = 8.5;
        S.legend_font = 7.0;
        S.annotation_font = 7.0;

        S.line_width = 1.20;
        S.marker_size = 4.0;
        S.marker_edge_width = 0.8;

        S.axes_line_width = 0.60;
        S.grid_alpha = 0.16;

        S.inset_axis_font = 6.5;
        S.inset_line_width = 1.00;
        S.inset_marker_size = 3.0;

      case {"half","compact","two-panel"}
        S.mode = 'half';
        S.fig_size = [4.60 3.20];

        S.axis_font = 15.0;         % -> about 5.2 pt printed, same as Fig. 3
        S.label_font = 16.0;
        S.legend_font = 14.0;
        S.annotation_font = 13.5;

        S.line_width = 2.3;
        S.marker_size = 7.5;
        S.marker_edge_width = 1.4;

        S.axes_line_width = 1.2;
        S.grid_alpha = 0.16;

        S.inset_axis_font = 9.0;
        S.inset_line_width = 1.35;
        S.inset_marker_size = 4.5;

    otherwise
        error(['Unknown plot_style mode "%s". Use "single" for one large ' ...
               'panel or "half" for two side-by-side panels.'],mode);
end

%% Apply to existing axes
if nargin >= 2 && ~isempty(ax) && isgraphics(ax,'axes')

    set(ax, ...
        'FontName',S.font_name, ...
        'FontSize',S.axis_font, ...
        'LineWidth',S.axes_line_width, ...
        'Box',S.box, ...
        'Layer','top', ...
        'TickLabelInterpreter',S.interpreter);

    if strcmpi(S.grid,'on')
        grid(ax,'on');
    else
        grid(ax,'off');
    end

    if strcmpi(S.minor_grid,'on')
        grid(ax,'minor');
    end

    if isprop(ax,'GridAlpha')
        ax.GridAlpha = S.grid_alpha;
    end

    if isgraphics(ax.XLabel)
        set(ax.XLabel,'FontName',S.font_name, ...
            'FontSize',S.label_font,'Interpreter',S.interpreter);
    end

    if isgraphics(ax.YLabel)
        set(ax.YLabel,'FontName',S.font_name, ...
            'FontSize',S.label_font,'Interpreter',S.interpreter);
    end

    if isgraphics(ax.ZLabel)
        set(ax.ZLabel,'FontName',S.font_name, ...
            'FontSize',S.label_font,'Interpreter',S.interpreter);
    end

    if isgraphics(ax.Title)
        ax.Title.String = '';
    end

    drawnow;
    try
        ti = ax.TightInset;
        pad = S.tight_margin;
        ax.LooseInset = max(ti,[pad pad pad pad]);
    catch
    end
end

end
