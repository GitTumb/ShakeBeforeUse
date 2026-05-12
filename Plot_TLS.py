import numpy as np
import matplotlib.pyplot as plt
import os
from  scipy.optimize import curve_fit

#Using costum plotting style, if not comment
plt.style.use("mystyle")

beta_list=[0.70]
Omega = 1.0
Omega_drive = 2.0
Lambda0 = 0.01
driving_type = 'gauss'
cost = True
plot_cost = False


def check_cost_file(filename, data):
    """
    Check that a result file contains cost data before plotting cost ratio.
    Params:
        filename: Loaded result filename.
        data: Loaded npz data object.
    Returns:
        None.
    """
    # Require both the filename tag and the saved arrays to be cost-enabled.
    if '_CYes' not in os.path.basename(filename):
        raise ValueError(
            f'Cannot plot cost ratio from file. '
            'Rerun TLS.py with cost = True to generate a CYes file.'
        )
    if 'cost_t' not in data.files or 'qfi_cost_ratio' not in data.files:
        raise ValueError(
            f'Cost values are missing from file. '
            'Rerun TLS.py with cost = True.'
        )

for beta in beta_list:
    #Build the result file path for this beta and drive.
    folder = os.path.join('Results', 'TLS', f'Omega_{Omega}',\
                          f'Beta_{beta:.2f}', 'Gauss', f'{Omega_drive}_{Lambda0}')
    cost_tag = 'CYes' if cost else 'CNo'
    filename = os.path.join(folder, f'qfi_beta_{beta}_{driving_type}_{cost_tag}.npz')

    #Load stored QFI traces and the equilibrium baseline.
    data = np.load(filename)
    t        = data['timepoints']
    full     = data['qfi_full']
    eig      = data['qfi_eig']
    F0       = data['baseline_F0']

    if plot_cost:
        check_cost_file(filename, data)
        cost_ratio = data['qfi_cost_ratio']
        xticks = [0,2*np.pi/Omega,4*np.pi/Omega]
        xtickslabel = ['0','2$\pi/\Omega$ ','4$\pi/\Omega$ ']

        #Plot the saved QFI-to-cost ratio.
        plt.figure()
        plt.plot(t, cost_ratio, label=r'$\mathcal{R}_2(t,\beta)$', lw=2)
        plt.xticks(xticks,labels=xtickslabel)
        plt.xlim(0,4*np.pi*Omega)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\mathcal{R}_2(t,\beta)$')
        plt.title(r'QFI-cost ratio: $\beta$ = '+str(beta)+'\,\,\,$\lambda_0 $ = '+\
                  str(Lambda0)+'\,\,\,$\omega_{drive} $ = '+str(Omega_drive))
        plt.legend()
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.grid(False)
        plt.savefig('qfi_cost_ratio.png',dpi=300,transparent=True)
        plt.show()
        continue

    #time-axis ticks in units of 2pi/Omega, so of period.
    xticks = [0,2*np.pi/Omega,4*np.pi/Omega]
    xtickslabel = ['0','2$\pi/\Omega$ ','4$\pi/\Omega$ ']
    #Plot the baseline, computed QFI traces, and fitted curve.
    plt.figure()
    plt.axhline(F0,   label=r'Equilibrium baseline', linestyle='-.',lw=2)
    plt.plot(t, full, label=r'Computed through Eq. 13',lw=2)
    plt.plot(t, eig,  label=r'Computed through eigenvalues of $\rho(t)$', linestyle='--')
    plt.xticks(xticks,labels=xtickslabel)
    plt.xlim(0,4*np.pi*Omega)
    plt.xlabel(r'$t$')
    plt.ylabel(r'QFI')
    plt.title(r'QFI vs time: $\beta$ = '+str(beta)+'\,\,\,$\lambda_0 $ = '+\
              str(Lambda0)+'\,\,\,$\omega_{drive} $ = '+str(Omega_drive))
    plt.legend()
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.grid(False)
    #Save and display the plot for this beta.
    plt.savefig('omegadrive_2.png',dpi=300,transparent=True)
    plt.show()
