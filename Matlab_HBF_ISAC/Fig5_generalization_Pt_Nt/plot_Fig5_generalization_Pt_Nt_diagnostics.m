clc
clear
close all

%% Fig. 5 DIAGNOSTICS: Pt/Nt generalization
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

c_ref = [0.15 0.15 0.15];
c_ssp = [0.00 0.45 0.74];
c_d50 = [0.85 0.33 0.10];
c_d200 = [0.93 0.69 0.13];

lw = 1.5;
ms = 5;

save_fig = true;
save_png = false;
save_pdf = false;

out_dir = fullfile(fig_dir,'results','figures_diagnostic');
if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

%% D1: gap with CI
fig1 = figure('Color','w','Name','Diagnostic gap','NumberTitle','off');
set(fig1,'Units','inches','Position',[1 1 7.1 3.0]);

subplot(1,2,1); hold on;
errorbar(Pt_dBm_list,Ptres.Gap_B1_50_avg,Ptres.Gap_B1_50_CI95,'-s', ...
    'Color',c_ssp,'LineWidth',lw,'MarkerSize',ms);
errorbar(Pt_dBm_list,Ptres.Gap_Deep50_avg,Ptres.Gap_Deep50_CI95,'-d', ...
    'Color',c_d50,'LineWidth',lw,'MarkerSize',ms);
errorbar(Pt_dBm_list,Ptres.Gap_Deep200_avg,Ptres.Gap_Deep200_CI95,'-o', ...
    'Color',c_d200,'LineWidth',lw,'MarkerSize',ms);
xline(Pt0_dBm,'k:','HandleVisibility','off');
grid on; box on;
xlabel('Transmit power, P_t (dBm)');
ylabel('Normalized remaining gap');
xticks(Pt_dBm_list);
legend({sprintf('SSP, N=%d',Nexe),sprintf('Deep, N=%d',Nexe), ...
    sprintf('Deep, N=%d',Ndeep)},'Location','northwest','Box','off');
set(gca,'FontName','Times New Roman','FontSize',9);

subplot(1,2,2); hold on;
errorbar(Nt_list,Ntres.Gap_B1_50_avg,Ntres.Gap_B1_50_CI95,'-s', ...
    'Color',c_ssp,'LineWidth',lw,'MarkerSize',ms);
errorbar(Nt_list,Ntres.Gap_Deep50_avg,Ntres.Gap_Deep50_CI95,'-d', ...
    'Color',c_d50,'LineWidth',lw,'MarkerSize',ms);
errorbar(Nt_list,Ntres.Gap_Deep200_avg,Ntres.Gap_Deep200_CI95,'-o', ...
    'Color',c_d200,'LineWidth',lw,'MarkerSize',ms);
xline(Nt0,'k:','HandleVisibility','off');
grid on; box on;
xlabel('Number of transmit antennas, N_t');
ylabel('Normalized remaining gap');
xticks(Nt_list);
legend({sprintf('SSP, N=%d',Nexe),sprintf('Deep, N=%d',Nexe), ...
    sprintf('Deep, N=%d',Ndeep)},'Location','northeast','Box','off');
set(gca,'FontName','Times New Roman','FontSize',9);

%% D2: physical SR/CRLB
fig2 = figure('Color','w','Name','Diagnostic physical performance','NumberTitle','off');
set(fig2,'Units','inches','Position',[1 1 7.1 3.0]);

subplot(1,2,1); hold on;
yyaxis left
h1=plot(Pt_dBm_list,Ptres.SRRef_avg,'-o','Color',c_ref,'LineWidth',lw);
h2=plot(Pt_dBm_list,Ptres.SR_B1_50_avg,'-s','Color',c_ssp,'LineWidth',lw);
h3=plot(Pt_dBm_list,Ptres.SR_Deep50_avg,'-d','Color',c_d50,'LineWidth',lw);
h4=plot(Pt_dBm_list,Ptres.SR_Deep200_avg,'-*','Color',c_d200,'LineWidth',lw);
ylabel('Sum rate (nat/s/Hz)');
yyaxis right
plot(Pt_dBm_list,Ptres.CRLBRef_avg,'--o','Color',c_ref,'LineWidth',lw,'HandleVisibility','off');
plot(Pt_dBm_list,Ptres.CRLB_B1_50_avg,'--s','Color',c_ssp,'LineWidth',lw,'HandleVisibility','off');
plot(Pt_dBm_list,Ptres.CRLB_Deep50_avg,'--d','Color',c_d50,'LineWidth',lw,'HandleVisibility','off');
plot(Pt_dBm_list,Ptres.CRLB_Deep200_avg,'--*','Color',c_d200,'LineWidth',lw,'HandleVisibility','off');
ylabel('CRLB');
xline(Pt0_dBm,'k:','HandleVisibility','off');
grid on; box on;
xlabel('Transmit power, P_t (dBm)');
legend([h1 h2 h3 h4],{'SSP ref',sprintf('SSP N=%d',Nexe), ...
    sprintf('Deep N=%d',Nexe),sprintf('Deep N=%d',Ndeep)}, ...
    'Location','best','Box','off');
set(gca,'FontName','Times New Roman','FontSize',9);

subplot(1,2,2); hold on;
yyaxis left
h1=plot(Nt_list,Ntres.SRRef_avg,'-o','Color',c_ref,'LineWidth',lw);
h2=plot(Nt_list,Ntres.SR_B1_50_avg,'-s','Color',c_ssp,'LineWidth',lw);
h3=plot(Nt_list,Ntres.SR_Deep50_avg,'-d','Color',c_d50,'LineWidth',lw);
h4=plot(Nt_list,Ntres.SR_Deep200_avg,'-*','Color',c_d200,'LineWidth',lw);
ylabel('Sum rate (nat/s/Hz)');
yyaxis right
plot(Nt_list,Ntres.CRLBRef_avg,'--o','Color',c_ref,'LineWidth',lw,'HandleVisibility','off');
plot(Nt_list,Ntres.CRLB_B1_50_avg,'--s','Color',c_ssp,'LineWidth',lw,'HandleVisibility','off');
plot(Nt_list,Ntres.CRLB_Deep50_avg,'--d','Color',c_d50,'LineWidth',lw,'HandleVisibility','off');
plot(Nt_list,Ntres.CRLB_Deep200_avg,'--*','Color',c_d200,'LineWidth',lw,'HandleVisibility','off');
ylabel('CRLB');
xline(Nt0,'k:','HandleVisibility','off');
grid on; box on;
xlabel('Number of transmit antennas, N_t');
legend([h1 h2 h3 h4],{'SSP ref',sprintf('SSP N=%d',Nexe), ...
    sprintf('Deep N=%d',Nexe),sprintf('Deep N=%d',Ndeep)}, ...
    'Location','best','Box','off');
set(gca,'FontName','Times New Roman','FontSize',9);

%% D3: acceleration gain
fig3 = figure('Color','w','Name','Diagnostic acceleration gain','NumberTitle','off');
set(fig3,'Units','inches','Position',[1 1 7.1 3.0]);

subplot(1,2,1); hold on;
errorbar(Pt_dBm_list,Ptres.Gain_Deep50_vs_B1_50_avg, ...
    Ptres.Gain_Deep50_vs_B1_50_CI95,'-o','Color',c_d50, ...
    'LineWidth',lw,'MarkerSize',ms);
yline(0,'k:'); xline(Pt0_dBm,'k:','HandleVisibility','off');
grid on; box on;
xlabel('Transmit power, P_t (dBm)');
ylabel('Gap reduction: SSP_{50} - Deep_{50}');
set(gca,'FontName','Times New Roman','FontSize',9);

subplot(1,2,2); hold on;
errorbar(Nt_list,Ntres.Gain_Deep50_vs_B1_50_avg, ...
    Ntres.Gain_Deep50_vs_B1_50_CI95,'-o','Color',c_d50, ...
    'LineWidth',lw,'MarkerSize',ms);
yline(0,'k:'); xline(Nt0,'k:','HandleVisibility','off');
grid on; box on;
xlabel('Number of transmit antennas, N_t');
ylabel('Gap reduction: SSP_{50} - Deep_{50}');
set(gca,'FontName','Times New Roman','FontSize',9);

%% D4: rejection counts
fig4 = figure('Color','w','Name','Diagnostic rejections','NumberTitle','off');
set(fig4,'Units','inches','Position',[1 1 7.1 3.0]);

subplot(1,2,1); hold on;
plot(Pt_dBm_list,Ptres.Reject_Deep50_avg,'-d','Color',c_d50,'LineWidth',lw);
plot(Pt_dBm_list,Ptres.Reject_Deep200_avg,'-o','Color',c_d200,'LineWidth',lw);
xline(Pt0_dBm,'k:','HandleVisibility','off');
grid on; box on;
xlabel('Transmit power, P_t (dBm)');
ylabel('Average rejected layers');
legend({sprintf('N=%d',Nexe),sprintf('N=%d',Ndeep)},'Location','best','Box','off');

subplot(1,2,2); hold on;
plot(Nt_list,Ntres.Reject_Deep50_avg,'-d','Color',c_d50,'LineWidth',lw);
plot(Nt_list,Ntres.Reject_Deep200_avg,'-o','Color',c_d200,'LineWidth',lw);
xline(Nt0,'k:','HandleVisibility','off');
grid on; box on;
xlabel('Number of transmit antennas, N_t');
ylabel('Average rejected layers');
legend({sprintf('N=%d',Nexe),sprintf('N=%d',Ndeep)},'Location','best','Box','off');

%% D5: reference iterations
fig5 = figure('Color','w','Name','Diagnostic reference iterations','NumberTitle','off');
set(fig5,'Units','inches','Position',[1 1 7.1 3.0]);

subplot(1,2,1);
plot(Pt_dBm_list,Ptres.RefIter_avg,'-o','Color',[0.2 0.2 0.2],'LineWidth',lw);
grid on; box on;
xlabel('Transmit power, P_t (dBm)');
ylabel('Average SSP reference iterations');
ylim([0 max(2000,1.02*max(Ptres.RefIter_avg))]);

subplot(1,2,2);
plot(Nt_list,Ntres.RefIter_avg,'-o','Color',[0.2 0.2 0.2],'LineWidth',lw);
grid on; box on;
xlabel('Number of transmit antennas, N_t');
ylabel('Average SSP reference iterations');
ylim([0 max(2000,1.02*max(Ntres.RefIter_avg))]);

%% Save
if save_fig
    savefig(fig1,fullfile(out_dir,'Fig5_diag_gap_CI.fig'));
    savefig(fig2,fullfile(out_dir,'Fig5_diag_physical_SR_CRLB.fig'));
    savefig(fig3,fullfile(out_dir,'Fig5_diag_acceleration_gain.fig'));
    savefig(fig4,fullfile(out_dir,'Fig5_diag_rejections.fig'));
    savefig(fig5,fullfile(out_dir,'Fig5_diag_reference_iterations.fig'));
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

if ~isfield(R,'Gain_Deep50_vs_B1_50_avg')
    gain = R.Gap_B1_50-R.Gap_Deep50;
    R.Gain_Deep50_vs_B1_50 = gain;
    R.Gain_Deep50_vs_B1_50_avg = mean(gain,2,'omitnan');
    nvalid = sum(isfinite(gain),2);
    R.Gain_Deep50_vs_B1_50_CI95 = ...
        1.96*std(gain,0,2,'omitnan')./sqrt(max(nvalid,1));
end
end
