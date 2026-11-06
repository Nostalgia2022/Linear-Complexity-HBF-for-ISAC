# Linear-Complexity-HBF-for-ISAC

MATLAB code for the paper
> T. Fang, N. T. Nguyen, and M. Juntti, "Linear-Complexity Hybrid Beamforming for ISAC: From Optimization to Deep Unfolding," submitted to *IEEE Transactions on Signal Processing*, 2026.
The code implements
SSP-HBF, a shifted successive projection algorithm for hybrid beamforming (HBF) that balances the multi-user sum rate (SR) and the multi-target Cramér–Rao lower bound (CRLB) with linear complexity in the number of transmit antennas, and
Deep-SSP-HBF, its deep-unfolded version with a few learned parameters per layer,
together with all scripts needed to reproduce the figures and the table of the paper.

Requirements
MATLAB (tested with R2026a)
Parallel Computing Toolbox (most simulation scripts use `parfor`)
Python with Jupyter, only for retraining Deep-SSP-HBF (the trained parameters are provided)

How to reproduce the results
Result files are not included because of their size. In each folder,
run the simulation script(s) to generate the result `.mat` files, and then
run the plotting script to reproduce the figure.
Keep `fixed_dataset_K8_Nt64_M3.mat`, `DeepSSP_depth_by_G_terminal_gap_g2max3_Fig_data.mat`, and `plot_style.m` in the main folder, since the scripts look for them there.
Fig. 3 and Table I (convergence and runtime)
`HBF_TableI_v2_full_vs_52_100ch.m`: SSP-HBF and Deep-SSP-HBF trajectories (G = 1, 2, 4, 8)
`HBF_TwoStage_G4_nominal_100ch.m`: two-stage baseline (fully digital design followed by one HBF decomposition)
`plot_Fig1_with_TwoStage_v2.m`: plots Fig. 3 and writes Table I
Fig. 4 (complexity)
`benchmark_SSP_HBF_complete_compact_vs_direct_CRLB_vs_Nt.m`, then `plot_compact_matrix_chain_vs_direct_Nt_v5.m`: Fig. 4(a)
`HBF_FLOPs_equal_performance_G4.m`, then `plot_Fig4b_FLOPs_equal_performance_v3.m`: Fig. 4(b)
Fig. 5 (SR–CRLB tradeoff)
`FD_SGPI_tradeoff_auto_parallel_serial_checkpoint_c2s_100ch.m`: fully digital benchmark
`HBF_SGPI_tradeoff_GC_parfor_checkpoint_100ch.m`: SSP-HBF for G = 1, 2, 4, 8
`HBF_DeepSSP_Fig2_tradeoff_GC_N125_supplement_100ch.m`: Deep-SSP-HBF with 125 layers
`plot_tradeoff_Digital_SSP_Deep125_G1_G2_G4_G8_v4.m`: plots Fig. 5
Figs. 6 and 7 (generalization)
Run the simulation script(s) in the corresponding folder, followed by the plotting script.
Citation
If you use this code, please cite
```bibtex
@article{Fang2026LinearHBF,
  author  = {Tianyu Fang and Nhan Thanh Nguyen and Markku Juntti},
  title   = {Linear-Complexity Hybrid Beamforming for {ISAC}: From Optimization to Deep Unfolding},
  journal = {submitted to IEEE Transactions on Signal Processing},
  year    = {2026}
}
```
This work builds on our previous fully digital design:
```bibtex
@article{Fang2026OptimalISAC,
  author  = {Tianyu Fang and Mengyuan Ma and Markku Juntti and Nir Shlezinger and A. Lee Swindlehurst and Nhan Thanh Nguyen},
  title   = {Optimal {ISAC} Beamforming Structure and Efficient Algorithms for Sum Rate and {CRLB} Balancing},
  journal = {IEEE Transactions on Signal Processing},
  volume  = {74},
  pages   = {2400--2416},
  year    = {2026},
  doi     = {10.1109/TSP.2026.3694198}
}
```
Contact
Tianyu Fang, Centre for Wireless Communications, University of Oulu, Finland
Email: tianyu.fang@oulu.fi
