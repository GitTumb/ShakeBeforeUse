import numpy as np
import matplotlib.pyplot as plt
from TLS import Evolve_QFI

#Using costum plotting style, if not comment
plt.style.use("mystyle")


##############################
#####Defaul Params & defs#####
##############################
beta_list   = np.linspace(0.1, 8.1, 20)

Omega = 1
Omega_drive = 0.99
Lambda0 = 0.2

# PROFILE 1
beta1 = 10   
sigma1 = 3      

# PROFILE 2
beta2 = 5
sigma2 = 3

# TIME DETAILS
dt = 1e-3
tfin = 4 * np.pi / Omega

SAVE = "QFIvsBeta.png"  # either None or plot name
SHOW = True 

#########################
###### Simulation #######
#########################

def shaped_gauss_run(beta: float, timepoints: np.ndarray, *, beta0_local: float, sigma_local: float):
    """
    Run one shaped Gaussian-drive QFI calculation.
    Params:
        beta: Inverse temperature.
        timepoints: Times where QFI values are evaluated.
        beta0_local: Gaussian envelope center.
        sigma_local: Gaussian envelope width.
    Returns:
        Tuple of (maximum QFI over time, equilibrium QFI, QFI time trace).
    """
    # Reuse the TLS QFI pipeline with a fixed Gaussian envelope.
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


def sweep_beta(betas, timepoints, *, beta0_local, sigma_local):
    """
    Sweep beta values for one Gaussian envelope.
    Params:
        betas: Inverse-temperature values to evaluate.
        timepoints: Times where QFI values are evaluated.
        beta0_local: Gaussian envelope center.
        sigma_local: Gaussian envelope width.
    Returns:
        Tuple of (maximum driven QFI values, equilibrium QFI values).
    """
    # Accumulate driven and equilibrium QFI values across beta.
    ys, ys_feq = [], []
    for b in betas:
        maxFQ, Feq, _ = shaped_gauss_run(b, timepoints, beta0_local=beta0_local, sigma_local=sigma_local)
        ys.append(maxFQ)
        ys_feq.append(Feq)
    return np.asarray(ys), np.asarray(ys_feq)

##########################
########## Main ##########
##########################

def main():
    """
    Build the beta sweep and plot the two shaped Gaussian profiles.
    """
    #Prepare beta and time grids.
    betas = beta_list  
    timepoints = np.arange(0.0, tfin + dt, dt)
    
    # Sweep the two profiles
    ys1, ys1_feq = sweep_beta(betas, timepoints, beta0_local=beta1, sigma_local=sigma1)
    ys2, ys2_feq = sweep_beta(betas, timepoints, beta0_local=beta2, sigma_local=sigma2)

    

    #plot
    fig, ax = plt.subplots()
    p1, = ax.plot(betas, ys1_feq, marker="s", lw=1, markersize=4, ls='--', label="Equilibrium")
    p2, = ax.plot(betas, ys1, marker="o", lw=1, markersize=4, ls='-.', label="Envelope 1")
    p3, = ax.plot(betas, ys2, marker="^", lw=1, markersize=4, ls='-.', label="Envelope 2")
    
    c1 = p1.get_color()
    c2 = p2.get_color()
    c3 = p3.get_color()

    x1, y1 = 4.5, 0.22
    ax.text(x1, y1,
        rf"$\beta_0$ = {beta1}",
        fontsize=9, color=c2, ha="left", va="bottom")
    ax.text(x1, y1-0.01,   # adjust 0.02 to your data scale
        rf"$s_\beta$ = {sigma1}",
        fontsize=9, color=c2, ha="left", va="top")

    x2,y2 = 7.0, 0.15
    ax.text(x2, y2,
        rf"$\beta_0$ = {beta2}",
        fontsize=9, color=c3, ha="left", va="bottom")
    ax.text(x2, y2-0.01,   # adjust 0.02 to your data scale
        rf"$s_\beta$ = {sigma2}",
        fontsize=9, color=c3, ha="left", va="top")


    ax.text(1, 0.05,   # adjust 0.02 to your data scale
        rf"Equilibrium baseline",
        fontsize=9, color=c1, ha="left", va="top")

    ax.set_xlim(0,8)

    ax.set_ylabel(r"$\max\limits_{t}\,\mathcal{F}_{\rho(t)}^\beta$")
    ax.set_xlabel(r"$\beta$")   
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if SAVE:
        # Save the plot if an output filename is configured.
        fig.savefig(SAVE, dpi=300, bbox_inches="tight")
    if SHOW:
        # Display the plot when requested.
        plt.show()


if __name__ == "__main__":
    # Configure direct script execution.
    SAVE = "QFIvsBeta.svg"
    SHOW = True
    main()
