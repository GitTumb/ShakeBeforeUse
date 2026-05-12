Purpose
=======

This repository contains supporting code for the paper:

Shake before use: universal enhancement of quantum thermometry by unitary driving
arXiv:2511.19631v3

The scripts implement and plot numerical examples for driven two-level-system
(TLS) thermometry. The main quantities are the quantum Fisher information (QFI),
its equilibrium baseline, driven enhancements, and an optional QFI-to-cost ratio.

Requirements
============

The scripts use:

- NumPy
- SciPy
- Matplotlib


Repository Contents
===================

TLS.py
------

Core simulation script. It contains the shared TLS definitions and the main QFI
pipeline used by the other scripts.

Main contents:

- Pauli matrices and thermal baseline construction.
- Linear, sinusoidal, and Gaussian-envelope drive functions.
- Unitary time evolution via solve_ivp.
- Evolve_QFI(...), the shared routine that computes:
  - qfi_full: QFI from the main SLD-based expression.
  - qfi_eig: QFI from the eigenbasis expression, evaluated every eig_step.
  - baseline_F0: equilibrium QFI.
  - optional cost_t and qfi_cost_ratio when compute_cost=True.

When run directly, TLS.py sweeps beta_list and saves one .npz result file per
beta under Results/TLS, particularly using this directory pattern:
Results/
  TLS/
    Omega_{Omega}/
      Beta_{beta:.2f}/
        {Drive}/
          {Omega_drive}_{Lambda0}/
            qfi_beta_{beta}_{driving_type}_{CYes_or_CNo}.npz

Example:
Results/TLS/Omega_1.0/Beta_0.70/Gauss/2.0_0.01/qfi_beta_0.7_gauss_CYes.npz

Drive folder names include:

- Gauss for driving_type = "gauss"
- Lin for driving_type = "lin"
- Cos for driving_type = "cos"

The CYes/CNo tag records whether cost quantities were saved:

- CYes: cost_t and qfi_cost_ratio are included.
- CNo: cost quantities are not included.

Each saved .npz file saved by TLS.py contains:

- timepoints: time grid.
- qfi_full: QFI from the main expression.
- qfi_eig: QFI from the eigenbasis expression.
- baseline_F0: equilibrium QFI.
- lambda0: drive strength used for the run.
- omega_drive: drive frequency used for the run.
- cost_computed: boolean recording whether cost output was enabled.
- beta0_used: Gaussian center, for Gaussian drives.
- sigma_used: Gaussian width, for Gaussian drives.

Only CYes files also contain:

- cost_t: cumulative integral of the squared drive amplitude.
- qfi_cost_ratio: (qfi_full - baseline_F0) / (cost_t + epsilon), where epsilon is used for numerical stability.

The QFI gain itself is not saved separately, and must be naturally computed
by means of the difference of qfi_full-baseline_F0



Plot_TLS.py
-----------

Loads saved .npz files from TLS.py and plots either:

- QFI versus time, when plot_cost=False.
- The QFI-to-cost ratio, when plot_cost=True.

The filename is chosen using the cost flag:

- cost=True loads a CYes file.
- cost=False loads a CNo file.

If plot_cost=True, the script checks that the selected file is a CYes file and
that cost_t and qfi_cost_ratio are present.


PeakShifting.py
---------------

Studies how the peak QFI changes as the true beta is varied for two fixed
Gaussian envelope profiles. 
When run directly, it:

- Defines two Gaussian profiles using beta0 and sigma.
- Sweeps beta values.
- Computes max_t QFI and the equilibrium baseline for each beta.
- Plots the two driven curves against the equilibrium baseline.


Plot_QFIvsMismatch.py
---------------------

Studies how the mismatch between the true inverse temperature beta_true and the
Gaussian-envelope mean beta0 effect the QFI increment, for several Gaussian widths sigma.




AI disclosure
=============

AI, specifically ChatGPT 4, has been used to help in the development and 
documentation of this code.
