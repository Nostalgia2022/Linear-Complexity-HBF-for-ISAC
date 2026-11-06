clc; clear;
%% Recompute direct-implementation FLOPs: explicit Nt x Nt matrices and
%% the shift (16) by power iteration on N*Omega2*N^H (no Nt^3 operation).
I_pow = 20;                               % power iterations per shift
file = fullfile('results','data_FLOPs_equal_performance_G4.mat');
D = load(file);  Res = D.Res;
K = 8; M = 3;

for i = 1:numel(Res)
    Nt = Res(i).Nt;  r = Res(i).raw;  F = Res(i).F;
    rN = K + 3*M;  m3 = 3*M;
    % objective of a candidate: Rx = W W^H and Gamma = Abar^H Rx Abar
    eval_d = 8*Nt^2*K + 8*m3*Nt^2 + 8*m3^2*Nt + 8*Nt*K*K ...
           + 60*M^2 + 2*(4*M)^3 + 10*K*K;
    % gradient: (H Sigma2 H^H) W and (Abar Sigma3 Abar^H) W
    grad_d = (8*Nt^2*K + 8*Nt^2*K + 8*Nt*K) ...
           + (8*Nt*m3^2 + 8*Nt^2*m3 + 8*Nt^2*K) ...
           + 2*(4*M)^3 + 40*M^3 + 100*M^2 ...
           + (8*Nt*rN^2 + 8*Nt^2*rN) ...          % form N*Omega2*N^H
           + I_pow*(8*Nt^2 + 8*Nt);                % power iteration
    per_it = grad_d + r.S_trials.*(F.norm + eval_d) ...
           + r.S_ao.*F.outer_full + r.S_inner.*F.analog_full;
    per_it(:,1) = 0;  per_it(r.S_trials==0) = 0;
    fl = zeros(numel(r.k_reach),1);
    for ch = 1:numel(r.k_reach)
        fl(ch) = sum(per_it(ch,1:r.k_reach(ch)));
    end
    Res(i).flops_dir = mean(fl);
    fprintf('Nt=%4d | GFLOPs: Deep %.3f, SSP %.3f, direct %.3f | direct/SSP %.1fx | direct/Deep %.1fx\n', ...
        Nt, Res(i).flops_deep/1e9, Res(i).flops_ssp/1e9, Res(i).flops_dir/1e9, ...
        Res(i).flops_dir/Res(i).flops_ssp, Res(i).flops_dir/Res(i).flops_deep);
end
D.Res = Res;
save(file,'-struct','D','-v7.3');