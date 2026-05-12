import numpy as np
import matplotlib.pyplot as plt
from TLS import Evolve_QFI

#Using costum plotting style, if not comment
plt.style.use("mystyle")

##############################
#####Defaul Params & defs#####
##############################
Omega = 1.0
Omega_drive = 1.0
Lambda0 = 0.01

dt = 1e-3
tfin = 4 * np.pi / Omega

SAVE = "Increment_vs_Mismatch.svg"
SHOW = True

#########################
###### Simulation #######
#########################

def get_MaxQFI(beta: float, timepoints: np.ndarray, *, beta0_local: float, sigma_local: float):
    """
    Run one fixed-envelope Gaussian QFI calculation.
    Params:
        beta: True inverse temperature.
        timepoints: Times where QFI values are evaluated.
        beta0_local: Gaussian envelope center.
        sigma_local: Gaussian envelope width.
    Returns:
        Tuple of (maximum QFI over time, equilibrium QFI, QFI time trace).
    """
    # Use the shared TLS evolution with a fixed Gaussian envelope.
    result = Evolve_QFI(
        beta,
        timepoints,
        lambda0=Lambda0,
        omega_drive=Omega_drive,
        driving_type='gauss',
        beta_ref=beta0_local,
        sigma_ref=sigma_local,
    )
    FQ = result['qfi_full']
    Feq = float(result['baseline_F0'])
    return float(np.max(FQ)), Feq, FQ


def sweep_mismatch(beta_true: float, beta0_list: np.ndarray, timepoints: np.ndarray, *, sigma_local: float):
    """
    Sweep Gaussian means and compute the QFI increment.
    Params:
        beta_true: True inverse temperature.
        beta0_list: Gaussian centers to evaluate.
        timepoints: Times where QFI values are evaluated.
        sigma_local: Gaussian envelope width.
    Returns:
        Array of increments max_t(F) - Feq for each Gaussian center.
    """
    
    ys_inc = []
    for b0 in beta0_list:
        maxFQ, Feq, _ = get_MaxQFI(
            beta_true,
            timepoints,
            beta0_local=b0,
            sigma_local=sigma_local,
        )
        ys_inc.append(maxFQ - Feq)
    return np.asarray(ys_inc, dtype=float)

##########################
########## Main ##########
##########################

def main():
    """
    Plot the QFI increment against Gaussian-center mismatch.
    Params:
        None.
    Returns:
        None.
    """
    # Build the time grid and beta0 offsets around the true beta.
    timepoints = np.arange(0.0, tfin + dt, dt)
    beta_true = 10.0
    dBeta_max = 3.0
    nB0 = 100
    delta_betas = np.linspace(-dBeta_max, dBeta_max, nB0)
    beta0_list = beta_true + delta_betas

    # Compare a few Gaussian envelope widths.
    sigma_list = [0.5, 1.5, 3.0] #progressively larger
    curves_inc = []
    for s in sigma_list:
        ys_inc = sweep_mismatch(beta_true, beta0_list, timepoints, sigma_local=s)
        curves_inc.append((s, ys_inc))

    fig, ax = plt.subplots()
    ax.text(
        0.02, 0.95,
        rf"$\beta^\ast = {beta_true}$",
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment='top',
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
        ),
    )

    for s, ys_inc in curves_inc:
        ax.plot(
            delta_betas,
            ys_inc,
            lw=1.6,
            marker="o",
            markersize=3,
            ls="-.",
            label=rf"$s_\beta={s}$",
        )

    ax.axhline(0.0, lw=1.0, ls="--")
    ax.set_xlabel(r"$\Delta\beta=\beta_0-\beta^\ast$")
    ax.set_ylabel(r"$\max\limits_{t}\,\mathcal{I}_{t}^{\beta^\ast}$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    if SAVE:
        fig.savefig(SAVE, dpi=300, bbox_inches="tight")
    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
