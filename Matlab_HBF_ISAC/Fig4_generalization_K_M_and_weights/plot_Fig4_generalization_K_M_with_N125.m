clc
clear
close all

%% ========================================================================
% Fig. 4 K/M GENERALIZATION AFTER N=125 SUPPLEMENT
%
% Plotting only. No simulation.
%
% Loads:
%   1) original completed K/M simulation:
%      data_Fig4_DeepSSP_generalization_K_M_G4_100ch.mat
%   2) N=125 supplement:
%      data_Fig4_DeepSSP_generalization_K_M_G4_N125_supplement_100ch.mat
%
% Generates:
%   Fig. A: normalized remaining gap
%           SSP N=50 / Deep N=50 / Deep N=125 / Deep N=200
%
%   Fig. B: clean physical-performance comparison
%           long-run SSP reference vs Deep N=125
%           solid = sum rate, dashed = CRLB
%
%   Fig. C: Deep-only depth progression
%           N=50 -> N=125 -> N=200
%
%   Fig. D: rejection-count diagnostic
%           Deep N=50 / N=125 / N=200
%% ========================================================================

%% Paths
script_fullpath = mfilename('fullpath');

if isempty(script_fullpath)
    fig_dir = pwd;
else
    fig_dir = fileparts(script_fullpath);
end

base_file = fullfile(fig_dir,'results', ...
    'data_Fig4_DeepSSP_generalization_K_M_G4_100ch.mat');

supp_file = fullfile(fig_dir,'results', ...
    'data_Fig4_DeepSSP_generalization_K_M_G4_N125_supplement_100ch.mat');

if ~exist(base_file,'file')
    error('Cannot find original Fig. 4 result:\n%s',base_file);
end

if ~exist(supp_file,'file')
    error('Cannot find N=125 supplement:\n%s',supp_file);
end

B = load(base_file);
S = load(supp_file);

required_base = {'K_list','M_list','Kres','Mres','K0','M0','G'};
for k = 1:numel(required_base)
    if ~isfield(B,required_base{k})
        error('Base result missing variable "%s".',required_base{k});
    end
end

required_supp = {'K_list','M_list','K125','M125','Nexe','G'};
for k = 1:numel(required_supp)
    if ~isfield(S,required_supp{k})
        error('Supplement missing variable "%s".',required_supp{k});
    end
end

K_list = double(B.K_list(:).');
M_list = double(B.M_list(:).');

if ~isequal(K_list,double(S.K_list(:).')) || ...
        ~isequal(M_list,double(S.M_list(:).'))
    error('K_list/M_list mismatch between base and N=125 supplement.');
end

if double(B.G) ~= double(S.G)
    error('G mismatch between base and supplement.');
end

Kres = ensure_averages(B.Kres);
Mres = ensure_averages(B.Mres);

K125 = ensure_averages(S.K125);
M125 = ensure_averages(S.M125);

K0 = double(B.K0);
M0 = double(B.M0);
G = double(B.G);

N50 = 50;
N125 = double(S.Nexe);
N200 = 200;

%% Plot settings
c_ssp  = [0.00 0.45 0.74];
c_d50  = [0.85 0.33 0.10];
c_d125 = [0.47 0.67 0.19];
c_d200 = [0.93 0.69 0.13];
c_ref  = [0.15 0.15 0.15];

lw = 1.6;
lw125 = 1.8;
ms = 5.0;

save_fig = true;
save_png = false;
save_pdf = false;

out_dir = fullfile(fig_dir,'results','figures_N125_KM');

if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

%% ========================================================================
% FIGURE A: normalized remaining gap
% This is the clearest generalization/finite-depth diagnostic.
%% ========================================================================
figA = figure('Color','w', ...
    'Name','K/M generalization gap with N125', ...
    'NumberTitle','off');

set(figA,'Units','inches','Position',[1 1 7.1 3.0]);

%% K panel
ax1 = subplot(1,2,1);
hold on;

errorbar(K_list,Kres.Gap_B1_50_avg,Kres.Gap_B1_50_CI95, ...
    '-s','Color',c_ssp,'LineWidth',lw, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

errorbar(K_list,Kres.Gap_Deep50_avg,Kres.Gap_Deep50_CI95, ...
    '-d','Color',c_d50,'LineWidth',lw, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

errorbar(K_list,K125.Gap_avg,K125.Gap_CI95, ...
    '-o','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

errorbar(K_list,Kres.Gap_Deep200_avg,Kres.Gap_Deep200_CI95, ...
    '-^','Color',c_d200,'LineWidth',lw, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

yl = ylim;
plot([K0 K0],yl,'k:','LineWidth',1.0,'HandleVisibility','off');

grid on;
box on;
xlabel('Number of communication users, K');
ylabel('Normalized remaining gap');
xticks(K_list);
xlim([min(K_list) max(K_list)]);

legend({sprintf('SSP, N=%d',N50), ...
        sprintf('Deep, N=%d',N50), ...
        sprintf('Deep, N=%d',N125), ...
        sprintf('Deep, N=%d',N200)}, ...
       'Location','best','FontSize',7.3,'Box','off');

text(0.04,0.93,'(a)', ...
    'Units','normalized', ...
    'FontName','Times New Roman', ...
    'FontSize',9,'FontWeight','bold');

set(ax1,'FontName','Times New Roman', ...
    'FontSize',8.8,'LineWidth',0.8,'Color','w');

%% M panel
ax2 = subplot(1,2,2);
hold on;

errorbar(M_list,Mres.Gap_B1_50_avg,Mres.Gap_B1_50_CI95, ...
    '-s','Color',c_ssp,'LineWidth',lw, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

errorbar(M_list,Mres.Gap_Deep50_avg,Mres.Gap_Deep50_CI95, ...
    '-d','Color',c_d50,'LineWidth',lw, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

errorbar(M_list,M125.Gap_avg,M125.Gap_CI95, ...
    '-o','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

errorbar(M_list,Mres.Gap_Deep200_avg,Mres.Gap_Deep200_CI95, ...
    '-^','Color',c_d200,'LineWidth',lw, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

yl = ylim;
plot([M0 M0],yl,'k:','LineWidth',1.0,'HandleVisibility','off');

grid on;
box on;
xlabel('Number of sensing targets, M');
ylabel('Normalized remaining gap');
xticks(M_list);
xlim([min(M_list) max(M_list)]);

legend({sprintf('SSP, N=%d',N50), ...
        sprintf('Deep, N=%d',N50), ...
        sprintf('Deep, N=%d',N125), ...
        sprintf('Deep, N=%d',N200)}, ...
       'Location','best','FontSize',7.3,'Box','off');

text(0.04,0.93,'(b)', ...
    'Units','normalized', ...
    'FontName','Times New Roman', ...
    'FontSize',9,'FontWeight','bold');

set(ax2,'FontName','Times New Roman', ...
    'FontSize',8.8,'LineWidth',0.8,'Color','w');

%% ========================================================================
% FIGURE B: clean physical-performance comparison
% SSP reference vs selected practical Deep N=125
%
% solid  = sum rate
% dashed = CRLB
%% ========================================================================
figB = figure('Color','w', ...
    'Name','K/M physical performance: SSP ref vs Deep N125', ...
    'NumberTitle','off');

set(figB,'Units','inches','Position',[1 1 7.1 3.0]);

%% K
ax3 = subplot(1,2,1);
hold on;

yyaxis left

h1 = plot(K_list,Kres.SRRef_avg, ...
    '-o','Color',c_ref,'LineWidth',1.75, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

h2 = plot(K_list,K125.SR_avg, ...
    '-d','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

ylabel('Sum rate (nat/s/Hz)');

yyaxis right

plot(K_list,Kres.CRLBRef_avg, ...
    '--o','Color',c_ref,'LineWidth',1.75, ...
    'MarkerSize',ms,'MarkerFaceColor','none', ...
    'HandleVisibility','off');

plot(K_list,K125.CRLB_avg, ...
    '--d','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms,'MarkerFaceColor','none', ...
    'HandleVisibility','off');

ylabel('CRLB');

grid on;
box on;
xlabel('Number of communication users, K');
xticks(K_list);
xlim([min(K_list) max(K_list)]);

legend([h1 h2], ...
    {'SSP reference',sprintf('Deep-SSP, N=%d',N125)}, ...
    'Location','best','FontSize',7.5,'Box','off');

text(0.04,0.07,'solid: SR   dashed: CRLB', ...
    'Units','normalized', ...
    'FontName','Times New Roman','FontSize',7.4);

text(0.04,0.93,'(a)', ...
    'Units','normalized', ...
    'FontName','Times New Roman', ...
    'FontSize',9,'FontWeight','bold');

ax3.YAxis(1).Color = [0 0 0];
ax3.YAxis(2).Color = [0 0 0];

set(ax3,'FontName','Times New Roman', ...
    'FontSize',8.8,'LineWidth',0.8,'Color','w');

%% M
ax4 = subplot(1,2,2);
hold on;

yyaxis left

h3 = plot(M_list,Mres.SRRef_avg, ...
    '-o','Color',c_ref,'LineWidth',1.75, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

h4 = plot(M_list,M125.SR_avg, ...
    '-d','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

ylabel('Sum rate (nat/s/Hz)');

yyaxis right

plot(M_list,Mres.CRLBRef_avg, ...
    '--o','Color',c_ref,'LineWidth',1.75, ...
    'MarkerSize',ms,'MarkerFaceColor','none', ...
    'HandleVisibility','off');

plot(M_list,M125.CRLB_avg, ...
    '--d','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms,'MarkerFaceColor','none', ...
    'HandleVisibility','off');

ylabel('CRLB');

grid on;
box on;
xlabel('Number of sensing targets, M');
xticks(M_list);
xlim([min(M_list) max(M_list)]);

legend([h3 h4], ...
    {'SSP reference',sprintf('Deep-SSP, N=%d',N125)}, ...
    'Location','best','FontSize',7.5,'Box','off');

text(0.04,0.07,'solid: SR   dashed: CRLB', ...
    'Units','normalized', ...
    'FontName','Times New Roman','FontSize',7.4);

text(0.04,0.93,'(b)', ...
    'Units','normalized', ...
    'FontName','Times New Roman', ...
    'FontSize',9,'FontWeight','bold');

ax4.YAxis(1).Color = [0 0 0];
ax4.YAxis(2).Color = [0 0 0];

set(ax4,'FontName','Times New Roman', ...
    'FontSize',8.8,'LineWidth',0.8,'Color','w');

%% ========================================================================
% FIGURE C: Deep-only depth progression
% Shows whether N=125 has already captured most of N=200.
%% ========================================================================
figC = figure('Color','w', ...
    'Name','Deep K/M depth progression', ...
    'NumberTitle','off');

set(figC,'Units','inches','Position',[1 1 7.1 3.0]);

subplot(1,2,1);
hold on;

plot(K_list,Kres.Gap_Deep50_avg, ...
    '--s','Color',c_d50,'LineWidth',lw, ...
    'MarkerSize',ms);

plot(K_list,K125.Gap_avg, ...
    '-o','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms);

plot(K_list,Kres.Gap_Deep200_avg, ...
    '-.^','Color',c_d200,'LineWidth',lw, ...
    'MarkerSize',ms);

xline(K0,'k:','HandleVisibility','off');

grid on;
box on;

xlabel('Number of communication users, K');
ylabel('Normalized remaining gap');
xticks(K_list);

legend({sprintf('Deep N=%d',N50), ...
        sprintf('Deep N=%d',N125), ...
        sprintf('Deep N=%d',N200)}, ...
       'Location','best','FontSize',7.5,'Box','off');

set(gca,'FontName','Times New Roman', ...
    'FontSize',9,'LineWidth',0.8);

subplot(1,2,2);
hold on;

plot(M_list,Mres.Gap_Deep50_avg, ...
    '--s','Color',c_d50,'LineWidth',lw, ...
    'MarkerSize',ms);

plot(M_list,M125.Gap_avg, ...
    '-o','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms);

plot(M_list,Mres.Gap_Deep200_avg, ...
    '-.^','Color',c_d200,'LineWidth',lw, ...
    'MarkerSize',ms);

xline(M0,'k:','HandleVisibility','off');

grid on;
box on;

xlabel('Number of sensing targets, M');
ylabel('Normalized remaining gap');
xticks(M_list);

legend({sprintf('Deep N=%d',N50), ...
        sprintf('Deep N=%d',N125), ...
        sprintf('Deep N=%d',N200)}, ...
       'Location','best','FontSize',7.5,'Box','off');

set(gca,'FontName','Times New Roman', ...
    'FontSize',9,'LineWidth',0.8);

%% ========================================================================
% FIGURE D: rejection-count diagnostic
%% ========================================================================
figD = figure('Color','w', ...
    'Name','Deep K/M rejection diagnostic', ...
    'NumberTitle','off');

set(figD,'Units','inches','Position',[1 1 7.1 3.0]);

subplot(1,2,1);
hold on;

plot(K_list,Kres.Reject_Deep50_avg, ...
    '--s','Color',c_d50,'LineWidth',lw,'MarkerSize',ms);

plot(K_list,K125.Reject_avg, ...
    '-o','Color',c_d125,'LineWidth',lw125,'MarkerSize',ms);

plot(K_list,Kres.Reject_Deep200_avg, ...
    '-.^','Color',c_d200,'LineWidth',lw,'MarkerSize',ms);

xline(K0,'k:','HandleVisibility','off');

grid on;
box on;

xlabel('Number of communication users, K');
ylabel('Average rejected layers');
xticks(K_list);

legend({sprintf('N=%d',N50),sprintf('N=%d',N125),sprintf('N=%d',N200)}, ...
    'Location','best','FontSize',7.5,'Box','off');

set(gca,'FontName','Times New Roman','FontSize',9,'LineWidth',0.8);

subplot(1,2,2);
hold on;

plot(M_list,Mres.Reject_Deep50_avg, ...
    '--s','Color',c_d50,'LineWidth',lw,'MarkerSize',ms);

plot(M_list,M125.Reject_avg, ...
    '-o','Color',c_d125,'LineWidth',lw125,'MarkerSize',ms);

plot(M_list,Mres.Reject_Deep200_avg, ...
    '-.^','Color',c_d200,'LineWidth',lw,'MarkerSize',ms);

xline(M0,'k:','HandleVisibility','off');

grid on;
box on;

xlabel('Number of sensing targets, M');
ylabel('Average rejected layers');
xticks(M_list);

legend({sprintf('N=%d',N50),sprintf('N=%d',N125),sprintf('N=%d',N200)}, ...
    'Location','best','FontSize',7.5,'Box','off');

set(gca,'FontName','Times New Roman','FontSize',9,'LineWidth',0.8);

%% ========================================================================
% Console summary
%% ========================================================================
fprintf('\n============================================================\n');
fprintf('K/M generalization with selected N=125\n');
fprintf('Training configuration: K=%d, M=%d, G=%d\n',K0,M0,G);
fprintf('============================================================\n');

fprintf('\nK sweep:\n');
fprintf(' K | SSP50 | Deep50 | Deep125 | Deep200\n');

for i = 1:numel(K_list)
    fprintf('%2d | %.5f | %.5f | %.5f | %.5f\n', ...
        K_list(i), ...
        Kres.Gap_B1_50_avg(i), ...
        Kres.Gap_Deep50_avg(i), ...
        K125.Gap_avg(i), ...
        Kres.Gap_Deep200_avg(i));
end

fprintf('\nM sweep:\n');
fprintf(' M | SSP50 | Deep50 | Deep125 | Deep200\n');

for i = 1:numel(M_list)
    fprintf('%2d | %.5f | %.5f | %.5f | %.5f\n', ...
        M_list(i), ...
        Mres.Gap_B1_50_avg(i), ...
        Mres.Gap_Deep50_avg(i), ...
        M125.Gap_avg(i), ...
        Mres.Gap_Deep200_avg(i));
end

fprintf('============================================================\n');

%% Save
if save_fig
    savefig(figA,fullfile(out_dir,'Fig4_KM_gap_N50_N125_N200.fig'));
    savefig(figB,fullfile(out_dir,'Fig4_KM_main_SSPref_vs_Deep125.fig'));
    savefig(figC,fullfile(out_dir,'Fig4_KM_Deep_depth_progression.fig'));
    savefig(figD,fullfile(out_dir,'Fig4_KM_rejection_diagnostic.fig'));
end

if save_png
    print(figA,fullfile(out_dir,'Fig4_KM_gap_N50_N125_N200.png'),'-dpng','-r300');
    print(figB,fullfile(out_dir,'Fig4_KM_main_SSPref_vs_Deep125.png'),'-dpng','-r300');
    print(figC,fullfile(out_dir,'Fig4_KM_Deep_depth_progression.png'),'-dpng','-r300');
    print(figD,fullfile(out_dir,'Fig4_KM_rejection_diagnostic.png'),'-dpng','-r300');
end

if save_pdf
    try
        print(figA,fullfile(out_dir,'Fig4_KM_gap_N50_N125_N200.pdf'),'-dpdf','-painters');
        print(figB,fullfile(out_dir,'Fig4_KM_main_SSPref_vs_Deep125.pdf'),'-dpdf','-painters');
        print(figC,fullfile(out_dir,'Fig4_KM_Deep_depth_progression.pdf'),'-dpdf','-painters');
        print(figD,fullfile(out_dir,'Fig4_KM_rejection_diagnostic.pdf'),'-dpdf','-painters');
    catch ME
        warning('PDF export failed: %s',ME.message);
    end
end

%% ========================================================================
% Local helper
%% ========================================================================
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
