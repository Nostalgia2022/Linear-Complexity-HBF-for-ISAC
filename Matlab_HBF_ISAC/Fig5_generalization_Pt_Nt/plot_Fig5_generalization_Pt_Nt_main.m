clc
clear
close all

%% Fig. 5 MAIN FIGURE: frozen Deep-SSP generalization over Pt and Nt
script_fullpath = mfilename('fullpath');
if isempty(script_fullpath)
    fig_dir = pwd;
else
    fig_dir = fileparts(script_fullpath);
end

data_file = fullfile(fig_dir,'results', ...
    'data_Fig5_DeepSSP_generalization_Pt_Nt_G4_100ch.mat');

if ~exist(data_file,'file')
    error('Cannot find Fig. 5 result file:\n%s',data_file);
end

S = load(data_file);

Pt_dBm_list = double(S.Pt_dBm_list(:).');
Nt_list = double(S.Nt_list(:).');
Ptres = ensure_averages(S.Ptres);
Ntres = ensure_averages(S.Ntres);

Pt0_dBm = double(S.Pt0_dBm);
Nt0 = 64;
Nexe = double(S.Nexe);
Ndeep = double(S.Ndeep);

c_ssp = [0.00 0.45 0.74];
c_d50 = [0.85 0.33 0.10];
c_d200 = [0.93 0.69 0.13];

lw = 1.65;
ms = 5.2;

save_fig = true;
save_png = false;
save_pdf = false;

out_dir = fullfile(fig_dir,'results','figures_main');
if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

fig = figure('Color','w','Name','Fig5 Pt/Nt generalization','NumberTitle','off');
set(fig,'Units','inches','Position',[1 1 7.1 3.0], ...
    'PaperUnits','inches','PaperPosition',[0 0 7.1 3.0], ...
    'PaperSize',[7.1 3.0]);

%% (a) Pt
ax1 = subplot(1,2,1);
hold(ax1,'on');

plot(Pt_dBm_list,Ptres.Gap_B1_50_avg,'-s', ...
    'Color',c_ssp,'LineWidth',lw,'MarkerSize',ms,'MarkerFaceColor','none');
plot(Pt_dBm_list,Ptres.Gap_Deep50_avg,'-d', ...
    'Color',c_d50,'LineWidth',lw,'MarkerSize',ms,'MarkerFaceColor','none');
plot(Pt_dBm_list,Ptres.Gap_Deep200_avg,'-o', ...
    'Color',c_d200,'LineWidth',lw,'MarkerSize',ms,'MarkerFaceColor','none');

yl = [0,1.08*max([Ptres.Gap_B1_50_avg(:);Ptres.Gap_Deep50_avg(:); ...
    Ptres.Gap_Deep200_avg(:);1e-3])];
ylim(yl);
plot([Pt0_dBm Pt0_dBm],yl,'k:','LineWidth',1.0,'HandleVisibility','off');

grid on; box on;
xlabel('Transmit power, P_t (dBm)');
ylabel('Normalized remaining gap');
xticks(Pt_dBm_list);
xlim([min(Pt_dBm_list) max(Pt_dBm_list)]);

legend({sprintf('SSP, N=%d',Nexe),sprintf('Deep-SSP, N=%d',Nexe), ...
    sprintf('Deep-SSP, N=%d',Ndeep)}, ...
    'Location','northwest','FontSize',7.5,'Box','off');

text(0.04,0.93,'(a)','Units','normalized','FontName','Times New Roman', ...
    'FontSize',9,'FontWeight','bold');
set(ax1,'FontName','Times New Roman','FontSize',8.8,'LineWidth',0.8,'Color','w');

%% (b) Nt
ax2 = subplot(1,2,2);
hold(ax2,'on');

plot(Nt_list,Ntres.Gap_B1_50_avg,'-s', ...
    'Color',c_ssp,'LineWidth',lw,'MarkerSize',ms,'MarkerFaceColor','none');
plot(Nt_list,Ntres.Gap_Deep50_avg,'-d', ...
    'Color',c_d50,'LineWidth',lw,'MarkerSize',ms,'MarkerFaceColor','none');
plot(Nt_list,Ntres.Gap_Deep200_avg,'-o', ...
    'Color',c_d200,'LineWidth',lw,'MarkerSize',ms,'MarkerFaceColor','none');

yl = [0,1.08*max([Ntres.Gap_B1_50_avg(:);Ntres.Gap_Deep50_avg(:); ...
    Ntres.Gap_Deep200_avg(:);1e-3])];
ylim(yl);
plot([Nt0 Nt0],yl,'k:','LineWidth',1.0,'HandleVisibility','off');

grid on; box on;
xlabel('Number of transmit antennas, N_t');
ylabel('Normalized remaining gap');
xticks(Nt_list);
xlim([min(Nt_list) max(Nt_list)]);

legend({sprintf('SSP, N=%d',Nexe),sprintf('Deep-SSP, N=%d',Nexe), ...
    sprintf('Deep-SSP, N=%d',Ndeep)}, ...
    'Location','northeast','FontSize',7.5,'Box','off');

text(0.04,0.93,'(b)','Units','normalized','FontName','Times New Roman', ...
    'FontSize',9,'FontWeight','bold');
set(ax2,'FontName','Times New Roman','FontSize',8.8,'LineWidth',0.8,'Color','w');

set(ax1,'Position',[0.085 0.18 0.38 0.76]);
set(ax2,'Position',[0.585 0.18 0.38 0.76]);

if save_fig
    savefig(fig,fullfile(out_dir,'Fig5_generalization_Pt_Nt_main.fig'));
end
if save_png
    print(fig,fullfile(out_dir,'Fig5_generalization_Pt_Nt_main.png'),'-dpng','-r300');
end
if save_pdf
    try
        print(fig,fullfile(out_dir,'Fig5_generalization_Pt_Nt_main.pdf'),'-dpdf','-painters');
    catch ME
        warning('PDF export failed: %s',ME.message);
    end
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
    if ~isfield(R,[name '_avg'])
        R.([name '_avg']) = mean(X,2,'omitnan');
    end
    if ~isfield(R,[name '_CI95'])
        nvalid = sum(isfinite(X),2);
        R.([name '_CI95']) = 1.96*std(X,0,2,'omitnan')./sqrt(max(nvalid,1));
    end
end
end
