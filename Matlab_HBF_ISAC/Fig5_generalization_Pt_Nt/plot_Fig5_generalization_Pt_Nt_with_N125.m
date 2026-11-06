clc
clear
close all

%% ========================================================================
% Fig. 5 Pt/Nt GENERALIZATION AFTER N=125 SUPPLEMENT
%
% Plotting only. No simulation.
%
% Loads:
%   1) original completed Pt/Nt simulation:
%      data_Fig5_DeepSSP_generalization_Pt_Nt_G4_100ch.mat
%   2) N=125 supplement:
%      data_Fig5_DeepSSP_generalization_Pt_Nt_G4_N125_supplement_100ch.mat
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
    'data_Fig5_DeepSSP_generalization_Pt_Nt_G4_100ch.mat');

supp_file = fullfile(fig_dir,'results', ...
    'data_Fig5_DeepSSP_generalization_Pt_Nt_G4_N125_supplement_100ch.mat');

if ~exist(base_file,'file')
    error('Cannot find original Fig. 5 result:\n%s',base_file);
end

if ~exist(supp_file,'file')
    error('Cannot find N=125 supplement:\n%s',supp_file);
end

B = load(base_file);
S = load(supp_file);

required_base = {'Pt_dBm_list','Nt_list','Ptres','Ntres','Pt0_dBm','G'};
for k = 1:numel(required_base)
    if ~isfield(B,required_base{k})
        error('Base result missing variable "%s".',required_base{k});
    end
end

required_supp = {'Pt_dBm_list','Nt_list','Pt125','Nt125','Nexe','G'};
for k = 1:numel(required_supp)
    if ~isfield(S,required_supp{k})
        error('Supplement missing variable "%s".',required_supp{k});
    end
end

Pt_dBm_list = double(B.Pt_dBm_list(:).');
Nt_list = double(B.Nt_list(:).');

if ~isequal(Pt_dBm_list,double(S.Pt_dBm_list(:).')) || ...
        ~isequal(Nt_list,double(S.Nt_list(:).'))
    error('Pt/Nt grids mismatch between base and N=125 supplement.');
end

if double(B.G) ~= double(S.G)
    error('G mismatch between base and supplement.');
end

Ptres = ensure_averages(B.Ptres);
Ntres = ensure_averages(B.Ntres);

Pt125 = ensure_averages(S.Pt125);
Nt125 = ensure_averages(S.Nt125);

Pt0_dBm = double(B.Pt0_dBm);
Nt0 = 64;
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

out_dir = fullfile(fig_dir,'results','figures_N125_PtNt');

if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

%% ========================================================================
% FIGURE A: normalized remaining gap
%% ========================================================================
figA = figure('Color','w', ...
    'Name','Pt/Nt generalization gap with N125', ...
    'NumberTitle','off');

set(figA,'Units','inches','Position',[1 1 7.1 3.0]);

%% Pt panel
ax1 = subplot(1,2,1);
hold on;

errorbar(Pt_dBm_list,Ptres.Gap_B1_50_avg,Ptres.Gap_B1_50_CI95, ...
    '-s','Color',c_ssp,'LineWidth',lw, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

errorbar(Pt_dBm_list,Ptres.Gap_Deep50_avg,Ptres.Gap_Deep50_CI95, ...
    '-d','Color',c_d50,'LineWidth',lw, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

errorbar(Pt_dBm_list,Pt125.Gap_avg,Pt125.Gap_CI95, ...
    '-o','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

errorbar(Pt_dBm_list,Ptres.Gap_Deep200_avg,Ptres.Gap_Deep200_CI95, ...
    '-^','Color',c_d200,'LineWidth',lw, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

yl = ylim;
plot([Pt0_dBm Pt0_dBm],yl,'k:','LineWidth',1.0,'HandleVisibility','off');

grid on;
box on;
xlabel('Transmit power, P_t (dBm)');
ylabel('Normalized remaining gap');
xticks(Pt_dBm_list);
xlim([min(Pt_dBm_list) max(Pt_dBm_list)]);

legend({sprintf('SSP, N=%d',N50), ...
        sprintf('Deep, N=%d',N50), ...
        sprintf('Deep, N=%d',N125), ...
        sprintf('Deep, N=%d',N200)}, ...
       'Location','northwest','FontSize',7.3,'Box','off');

text(0.04,0.93,'(a)', ...
    'Units','normalized', ...
    'FontName','Times New Roman', ...
    'FontSize',9,'FontWeight','bold');

set(ax1,'FontName','Times New Roman', ...
    'FontSize',8.8,'LineWidth',0.8,'Color','w');

%% Nt panel
ax2 = subplot(1,2,2);
hold on;

errorbar(Nt_list,Ntres.Gap_B1_50_avg,Ntres.Gap_B1_50_CI95, ...
    '-s','Color',c_ssp,'LineWidth',lw, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

errorbar(Nt_list,Ntres.Gap_Deep50_avg,Ntres.Gap_Deep50_CI95, ...
    '-d','Color',c_d50,'LineWidth',lw, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

errorbar(Nt_list,Nt125.Gap_avg,Nt125.Gap_CI95, ...
    '-o','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

errorbar(Nt_list,Ntres.Gap_Deep200_avg,Ntres.Gap_Deep200_CI95, ...
    '-^','Color',c_d200,'LineWidth',lw, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

yl = ylim;
plot([Nt0 Nt0],yl,'k:','LineWidth',1.0,'HandleVisibility','off');

grid on;
box on;
xlabel('Number of transmit antennas, N_t');
ylabel('Normalized remaining gap');
xticks(Nt_list);
xlim([min(Nt_list) max(Nt_list)]);

legend({sprintf('SSP, N=%d',N50), ...
        sprintf('Deep, N=%d',N50), ...
        sprintf('Deep, N=%d',N125), ...
        sprintf('Deep, N=%d',N200)}, ...
       'Location','northeast','FontSize',7.3,'Box','off');

text(0.04,0.93,'(b)', ...
    'Units','normalized', ...
    'FontName','Times New Roman', ...
    'FontSize',9,'FontWeight','bold');

set(ax2,'FontName','Times New Roman', ...
    'FontSize',8.8,'LineWidth',0.8,'Color','w');

%% ========================================================================
% FIGURE B: clean physical-performance comparison
% SSP reference vs selected practical Deep N=125
%% ========================================================================
figB = figure('Color','w', ...
    'Name','Pt/Nt physical performance: SSP ref vs Deep N125', ...
    'NumberTitle','off');

set(figB,'Units','inches','Position',[1 1 7.1 3.0]);

%% Pt
ax3 = subplot(1,2,1);
hold on;

yyaxis left

h1 = plot(Pt_dBm_list,Ptres.SRRef_avg, ...
    '-o','Color',c_ref,'LineWidth',1.75, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

h2 = plot(Pt_dBm_list,Pt125.SR_avg, ...
    '-d','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

ylabel('Sum rate (nat/s/Hz)');

yyaxis right

plot(Pt_dBm_list,Ptres.CRLBRef_avg, ...
    '--o','Color',c_ref,'LineWidth',1.75, ...
    'MarkerSize',ms,'MarkerFaceColor','none', ...
    'HandleVisibility','off');

plot(Pt_dBm_list,Pt125.CRLB_avg, ...
    '--d','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms,'MarkerFaceColor','none', ...
    'HandleVisibility','off');

ylabel('CRLB');

grid on;
box on;
xlabel('Transmit power, P_t (dBm)');
xticks(Pt_dBm_list);
xlim([min(Pt_dBm_list) max(Pt_dBm_list)]);

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

%% Nt
ax4 = subplot(1,2,2);
hold on;

yyaxis left

h3 = plot(Nt_list,Ntres.SRRef_avg, ...
    '-o','Color',c_ref,'LineWidth',1.75, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

h4 = plot(Nt_list,Nt125.SR_avg, ...
    '-d','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms,'MarkerFaceColor','none');

ylabel('Sum rate (nat/s/Hz)');

yyaxis right

plot(Nt_list,Ntres.CRLBRef_avg, ...
    '--o','Color',c_ref,'LineWidth',1.75, ...
    'MarkerSize',ms,'MarkerFaceColor','none', ...
    'HandleVisibility','off');

plot(Nt_list,Nt125.CRLB_avg, ...
    '--d','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms,'MarkerFaceColor','none', ...
    'HandleVisibility','off');

ylabel('CRLB');

grid on;
box on;
xlabel('Number of transmit antennas, N_t');
xticks(Nt_list);
xlim([min(Nt_list) max(Nt_list)]);

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
%% ========================================================================
figC = figure('Color','w', ...
    'Name','Deep Pt/Nt depth progression', ...
    'NumberTitle','off');

set(figC,'Units','inches','Position',[1 1 7.1 3.0]);

subplot(1,2,1);
hold on;

plot(Pt_dBm_list,Ptres.Gap_Deep50_avg, ...
    '--s','Color',c_d50,'LineWidth',lw, ...
    'MarkerSize',ms);

plot(Pt_dBm_list,Pt125.Gap_avg, ...
    '-o','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms);

plot(Pt_dBm_list,Ptres.Gap_Deep200_avg, ...
    '-.^','Color',c_d200,'LineWidth',lw, ...
    'MarkerSize',ms);

xline(Pt0_dBm,'k:','HandleVisibility','off');

grid on;
box on;

xlabel('Transmit power, P_t (dBm)');
ylabel('Normalized remaining gap');
xticks(Pt_dBm_list);

legend({sprintf('Deep N=%d',N50), ...
        sprintf('Deep N=%d',N125), ...
        sprintf('Deep N=%d',N200)}, ...
       'Location','best','FontSize',7.5,'Box','off');

set(gca,'FontName','Times New Roman', ...
    'FontSize',9,'LineWidth',0.8);

subplot(1,2,2);
hold on;

plot(Nt_list,Ntres.Gap_Deep50_avg, ...
    '--s','Color',c_d50,'LineWidth',lw, ...
    'MarkerSize',ms);

plot(Nt_list,Nt125.Gap_avg, ...
    '-o','Color',c_d125,'LineWidth',lw125, ...
    'MarkerSize',ms);

plot(Nt_list,Ntres.Gap_Deep200_avg, ...
    '-.^','Color',c_d200,'LineWidth',lw, ...
    'MarkerSize',ms);

xline(Nt0,'k:','HandleVisibility','off');

grid on;
box on;

xlabel('Number of transmit antennas, N_t');
ylabel('Normalized remaining gap');
xticks(Nt_list);

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
    'Name','Deep Pt/Nt rejection diagnostic', ...
    'NumberTitle','off');

set(figD,'Units','inches','Position',[1 1 7.1 3.0]);

subplot(1,2,1);
hold on;

plot(Pt_dBm_list,Ptres.Reject_Deep50_avg, ...
    '--s','Color',c_d50,'LineWidth',lw,'MarkerSize',ms);

plot(Pt_dBm_list,Pt125.Reject_avg, ...
    '-o','Color',c_d125,'LineWidth',lw125,'MarkerSize',ms);

plot(Pt_dBm_list,Ptres.Reject_Deep200_avg, ...
    '-.^','Color',c_d200,'LineWidth',lw,'MarkerSize',ms);

xline(Pt0_dBm,'k:','HandleVisibility','off');

grid on;
box on;

xlabel('Transmit power, P_t (dBm)');
ylabel('Average rejected layers');
xticks(Pt_dBm_list);

legend({sprintf('N=%d',N50),sprintf('N=%d',N125),sprintf('N=%d',N200)}, ...
    'Location','best','FontSize',7.5,'Box','off');

set(gca,'FontName','Times New Roman','FontSize',9,'LineWidth',0.8);

subplot(1,2,2);
hold on;

plot(Nt_list,Ntres.Reject_Deep50_avg, ...
    '--s','Color',c_d50,'LineWidth',lw,'MarkerSize',ms);

plot(Nt_list,Nt125.Reject_avg, ...
    '-o','Color',c_d125,'LineWidth',lw125,'MarkerSize',ms);

plot(Nt_list,Ntres.Reject_Deep200_avg, ...
    '-.^','Color',c_d200,'LineWidth',lw,'MarkerSize',ms);

xline(Nt0,'k:','HandleVisibility','off');

grid on;
box on;

xlabel('Number of transmit antennas, N_t');
ylabel('Average rejected layers');
xticks(Nt_list);

legend({sprintf('N=%d',N50),sprintf('N=%d',N125),sprintf('N=%d',N200)}, ...
    'Location','best','FontSize',7.5,'Box','off');

set(gca,'FontName','Times New Roman','FontSize',9,'LineWidth',0.8);

%% ========================================================================
% Console summary
%% ========================================================================
fprintf('\n============================================================\n');
fprintf('Pt/Nt generalization with selected N=125\n');
fprintf('Training configuration: Pt=%g dBm, Nt=%d, G=%d\n', ...
    Pt0_dBm,Nt0,G);
fprintf('============================================================\n');

fprintf('\nPt sweep:\n');
fprintf(' Pt | SSP50 | Deep50 | Deep125 | Deep200\n');

for i = 1:numel(Pt_dBm_list)
    fprintf('%3.0f | %.5f | %.5f | %.5f | %.5f\n', ...
        Pt_dBm_list(i), ...
        Ptres.Gap_B1_50_avg(i), ...
        Ptres.Gap_Deep50_avg(i), ...
        Pt125.Gap_avg(i), ...
        Ptres.Gap_Deep200_avg(i));
end

fprintf('\nNt sweep:\n');
fprintf(' Nt | SSP50 | Deep50 | Deep125 | Deep200\n');

for i = 1:numel(Nt_list)
    fprintf('%3d | %.5f | %.5f | %.5f | %.5f\n', ...
        Nt_list(i), ...
        Ntres.Gap_B1_50_avg(i), ...
        Ntres.Gap_Deep50_avg(i), ...
        Nt125.Gap_avg(i), ...
        Ntres.Gap_Deep200_avg(i));
end

fprintf('============================================================\n');

%% Save
if save_fig
    savefig(figA,fullfile(out_dir,'Fig5_PtNt_gap_N50_N125_N200.fig'));
    savefig(figB,fullfile(out_dir,'Fig5_PtNt_main_SSPref_vs_Deep125.fig'));
    savefig(figC,fullfile(out_dir,'Fig5_PtNt_Deep_depth_progression.fig'));
    savefig(figD,fullfile(out_dir,'Fig5_PtNt_rejection_diagnostic.fig'));
end

if save_png
    print(figA,fullfile(out_dir,'Fig5_PtNt_gap_N50_N125_N200.png'),'-dpng','-r300');
    print(figB,fullfile(out_dir,'Fig5_PtNt_main_SSPref_vs_Deep125.png'),'-dpng','-r300');
    print(figC,fullfile(out_dir,'Fig5_PtNt_Deep_depth_progression.png'),'-dpng','-r300');
    print(figD,fullfile(out_dir,'Fig5_PtNt_rejection_diagnostic.png'),'-dpng','-r300');
end

if save_pdf
    try
        print(figA,fullfile(out_dir,'Fig5_PtNt_gap_N50_N125_N200.pdf'),'-dpdf','-painters');
        print(figB,fullfile(out_dir,'Fig5_PtNt_main_SSPref_vs_Deep125.pdf'),'-dpdf','-painters');
        print(figC,fullfile(out_dir,'Fig5_PtNt_Deep_depth_progression.pdf'),'-dpdf','-painters');
        print(figD,fullfile(out_dir,'Fig5_PtNt_rejection_diagnostic.pdf'),'-dpdf','-painters');
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
