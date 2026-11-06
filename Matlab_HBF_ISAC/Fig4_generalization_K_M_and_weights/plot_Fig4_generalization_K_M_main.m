clc
clear
close all

%% ========================================================================
% Fig. 4 MAIN PAPER FIGURE (plotting only)
%
% Frozen-model generalization over:
%   (a) number of communication users K
%   (b) number of sensing targets M
%
% Main comparison:
%   - Long-run SSP-HBF reference
%   - Conventional SSP-HBF, N = 50
%   - Deep-SSP-HBF, N = 50
%
% Solid lines : sum rate
% Dashed lines: CRLB
%
% Deep N=200 is intentionally omitted from the main paper figure and is
% reserved for diagnostics, because N=50 is the practical deployment depth.
%% ========================================================================

%% Paths
script_fullpath = mfilename('fullpath');

if isempty(script_fullpath)
    fig_dir = pwd;
else
    fig_dir = fileparts(script_fullpath);
end

data_file = fullfile(fig_dir,'results', ...
    'data_Fig4_DeepSSP_generalization_K_M_G4_100ch.mat');

if ~exist(data_file,'file')
    error('Cannot find Fig. 4 result file:\n%s',data_file);
end

S = load(data_file);

required = {'K_list','M_list','Kres','Mres','K0','M0','G','Nexe'};
for kk = 1:numel(required)
    if ~isfield(S,required{kk})
        error('Result file missing variable "%s".',required{kk});
    end
end

K_list = double(S.K_list(:).');
M_list = double(S.M_list(:).');

Kres = ensure_averages(S.Kres);
Mres = ensure_averages(S.Mres);

K0 = double(S.K0);
M0 = double(S.M0);
G = double(S.G);
Nexe = double(S.Nexe);

%% Plot settings
c_ref  = [0.15 0.15 0.15];
c_ssp  = [0.00 0.45 0.74];
c_deep = [0.85 0.33 0.10];

lw_ref  = 1.65;
lw_ssp  = 1.55;
lw_deep = 1.70;
ms = 5.0;

save_fig = true;
save_png = false;
save_pdf = false;

out_dir = fullfile(fig_dir,'results','figures_main');
if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

%% Main two-panel figure
fig = figure('Color','w', ...
    'Name','Fig4 K/M generalization', ...
    'NumberTitle','off');

set(fig, ...
    'Units','inches', ...
    'Position',[1 1 7.1 3.05], ...
    'PaperUnits','inches', ...
    'PaperPosition',[0 0 7.1 3.05], ...
    'PaperSize',[7.1 3.05]);

%% (a) Generalization over K
ax1 = subplot(1,2,1);
hold(ax1,'on');

yyaxis(ax1,'left');

h1 = plot(K_list,Kres.SRRef_avg, ...
    '-o','Color',c_ref,'LineWidth',lw_ref, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

h2 = plot(K_list,Kres.SR_B1_50_avg, ...
    '-s','Color',c_ssp,'LineWidth',lw_ssp, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

h3 = plot(K_list,Kres.SR_Deep50_avg, ...
    '-d','Color',c_deep,'LineWidth',lw_deep, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

ylabel('Sum rate (nat/s/Hz)');

yyaxis(ax1,'right');

plot(K_list,Kres.CRLBRef_avg, ...
    '--o','Color',c_ref,'LineWidth',lw_ref, ...
    'MarkerSize',ms,'MarkerFaceColor','none', ...
    'HandleVisibility','off');

plot(K_list,Kres.CRLB_B1_50_avg, ...
    '--s','Color',c_ssp,'LineWidth',lw_ssp, ...
    'MarkerSize',ms,'MarkerFaceColor','none', ...
    'HandleVisibility','off');

plot(K_list,Kres.CRLB_Deep50_avg, ...
    '--d','Color',c_deep,'LineWidth',lw_deep, ...
    'MarkerSize',ms,'MarkerFaceColor','none', ...
    'HandleVisibility','off');

ylabel('CRLB');
xlabel('Number of communication users, K');

grid on;
box on;
xlim([min(K_list) max(K_list)]);
xticks(K_list);

legend([h1 h2 h3], ...
    {'SSP reference',sprintf('SSP, N=%d',Nexe), ...
     sprintf('Deep-SSP, N=%d',Nexe)}, ...
    'Location','best','FontSize',7.3,'Box','off');

text(0.04,0.07,'solid: SR   dashed: CRLB', ...
    'Units','normalized', ...
    'FontName','Times New Roman', ...
    'FontSize',7.5, ...
    'Color',[0.25 0.25 0.25]);

text(0.04,0.93,'(a)', ...
    'Units','normalized', ...
    'FontName','Times New Roman', ...
    'FontSize',9, ...
    'FontWeight','bold');

ax1.YAxis(1).Color = [0 0 0];
ax1.YAxis(2).Color = [0 0 0];

set(ax1, ...
    'FontName','Times New Roman', ...
    'FontSize',8.8, ...
    'LineWidth',0.8, ...
    'Color','w');

%% (b) Generalization over M
ax2 = subplot(1,2,2);
hold(ax2,'on');

yyaxis(ax2,'left');

h4 = plot(M_list,Mres.SRRef_avg, ...
    '-o','Color',c_ref,'LineWidth',lw_ref, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

h5 = plot(M_list,Mres.SR_B1_50_avg, ...
    '-s','Color',c_ssp,'LineWidth',lw_ssp, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

h6 = plot(M_list,Mres.SR_Deep50_avg, ...
    '-d','Color',c_deep,'LineWidth',lw_deep, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

ylabel('Sum rate (nat/s/Hz)');

yyaxis(ax2,'right');

plot(M_list,Mres.CRLBRef_avg, ...
    '--o','Color',c_ref,'LineWidth',lw_ref, ...
    'MarkerSize',ms,'MarkerFaceColor','none', ...
    'HandleVisibility','off');

plot(M_list,Mres.CRLB_B1_50_avg, ...
    '--s','Color',c_ssp,'LineWidth',lw_ssp, ...
    'MarkerSize',ms,'MarkerFaceColor','none', ...
    'HandleVisibility','off');

plot(M_list,Mres.CRLB_Deep50_avg, ...
    '--d','Color',c_deep,'LineWidth',lw_deep, ...
    'MarkerSize',ms,'MarkerFaceColor','none', ...
    'HandleVisibility','off');

ylabel('CRLB');
xlabel('Number of sensing targets, M');

grid on;
box on;
xlim([min(M_list) max(M_list)]);
xticks(M_list);

legend([h4 h5 h6], ...
    {'SSP reference',sprintf('SSP, N=%d',Nexe), ...
     sprintf('Deep-SSP, N=%d',Nexe)}, ...
    'Location','best','FontSize',7.3,'Box','off');

text(0.04,0.07,'solid: SR   dashed: CRLB', ...
    'Units','normalized', ...
    'FontName','Times New Roman', ...
    'FontSize',7.5, ...
    'Color',[0.25 0.25 0.25]);

text(0.04,0.93,'(b)', ...
    'Units','normalized', ...
    'FontName','Times New Roman', ...
    'FontSize',9, ...
    'FontWeight','bold');

ax2.YAxis(1).Color = [0 0 0];
ax2.YAxis(2).Color = [0 0 0];

set(ax2, ...
    'FontName','Times New Roman', ...
    'FontSize',8.8, ...
    'LineWidth',0.8, ...
    'Color','w');

%% Compact subplot spacing
set(ax1,'Position',[0.085 0.18 0.37 0.76]);
set(ax2,'Position',[0.585 0.18 0.37 0.76]);

fprintf('\n============================================================\n');
fprintf('Fig. 4 main figure plotted.\n');
fprintf('Training configuration: K=%d, M=%d, G=%d\n',K0,M0,G);
fprintf('Practical execution depth: N=%d\n',Nexe);
fprintf('============================================================\n');

%% Optional save
if save_fig
    savefig(fig,fullfile(out_dir,'Fig4_generalization_K_M_main.fig'));
end

if save_png
    print(fig,fullfile(out_dir,'Fig4_generalization_K_M_main.png'), ...
        '-dpng','-r300');
end

if save_pdf
    try
        print(fig,fullfile(out_dir,'Fig4_generalization_K_M_main.pdf'), ...
            '-dpdf','-painters');
    catch ME
        warning('PDF export failed: %s',ME.message);
    end
end

%% Local helper
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
        ci_name = [name '_CI95'];

        if ~isfield(R,avg_name)
            R.(avg_name) = mean(X,2,'omitnan');
        end

        if ~isfield(R,ci_name)
            nvalid = sum(isfinite(X),2);
            R.(ci_name) = ...
                1.96*std(X,0,2,'omitnan') ./ sqrt(max(nvalid,1));
        end
    end
end
