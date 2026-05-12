import os
import numpy as np
from scipy.linalg import solve_sylvester, eigh, polar
from scipy.integrate import solve_ivp, cumtrapz
from functools import partial

##############################
#####Defaul Params & defs#####
##############################

hbar = 1.0
Omega = 1.0
Omega_drive = 2.0
Lambda0 = 0.01
Tau = 1/(2*np.pi)
SEED = 28092 #RNG

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

###########################
#########Utilities#########
###########################

def commutator(A, B):
    """
    Return the matrix commutator [A, B].
    Params:
        A: Left matrix.
        B: Right matrix.
    Returns:
        Matrix A @ B - B @ A.
    """
    return A @ B - B @ A

# Thermal baseline
def thermal(beta):
    """
    Compute the initial thermal state and baseline Fisher quantities.
    Params:
        beta: Inverse temperature.
    Returns:
        Tuple of (pi0, d_pi0, L0, F0).
    """
    # Build and diagonalize the static two-level Hamiltonian.
    H0 = 0.5 * hbar * Omega * sz
    E, P = eigh(H0)

    # Construct the normalized Gibbs state.
    weights = np.exp(-beta * (E - E.min()))
    pi0 = P @ np.diag(weights) @ P.conj().T
    pi0 /= np.trace(pi0)

    # Compute beta derivative, analytic SLD, and baseline QFI.
    d_pi0 = (-H0 + np.trace(pi0 @ H0) * I2) @ pi0
    L0 = -H0 + np.trace(pi0 @ H0) * I2
    F0 = np.real(np.trace(pi0 @ L0 @ L0))
    return pi0, d_pi0, L0, F0

#Drives
def linear_lambda(t, beta, lambda0):
    """
    Return the linear drive amplitude.
    Params:
        t: Time.
        beta: Inverse temperature.
        lambda0: Drive strength.
    Returns:
        Linear drive amplitude at time t.
    """
    return lambda0 * beta * t / Tau

def linear_lambda_dot(t, beta, lambda0):
    """
    Return the beta derivative of the linear drive amplitude.
    Params:
        t: Time.
        beta: Inverse temperature.
        lambda0: Drive strength.
    Returns:
        Linear drive derivative at time t.
    """
    return (lambda0 * t) / Tau

def H_lin(t, beta, lambda0):
    """
    Build the Hamiltonian for the linear drive.
    Params:
        t: Time.
        beta: Inverse temperature.
        lambda0: Drive strength.
    Returns:
        Driven two-level Hamiltonian.
    """
    return 0.5*hbar*Omega*sz + linear_lambda(t, beta, lambda0=lambda0) * sx

def sinusoidal_lambda(t, beta, lambda0, omega_drive):
    """
    Return the sinusoidal drive amplitude.
    Params:
        t: Time.
        beta: Inverse temperature.
        lambda0: Drive strength.
        omega_drive: Drive frequency.
    Returns:
        Sinusoidal drive amplitude at time t.
    """
    return lambda0 * beta * np.cos(omega_drive * t)

def sinusoidal_lambda_dot(t, beta, lambda0, omega_drive):
    """
    Return the beta derivative of the sinusoidal drive amplitude.
    Params:
        t: Time.
        beta: Inverse temperature.
        lambda0: Drive strength.
        omega_drive: Drive frequency.
    Returns:
        Sinusoidal drive derivative at time t.
    """
    return lambda0 * np.cos(omega_drive * t)

def H_sin(t, beta, lambda0, omega_drive):
    """
    Build the Hamiltonian for the sinusoidal drive.
    Params:
        t: Time.
        beta: Inverse temperature.
        lambda0: Drive strength.
        omega_drive: Drive frequency.
    Returns:
        Driven two-level Hamiltonian.
    """
    return 0.5*hbar*Omega*sz + sinusoidal_lambda(
        t, beta, lambda0=lambda0, omega_drive=omega_drive
    ) * sx

def gaussian_envelope(beta, beta0, sigma):
    """
    Evaluate the Gaussian beta envelope.
    Params:
        beta: Inverse temperature.
        beta0: Gaussian center.
        sigma: Gaussian width.
    Returns:
        Envelope value at beta.
    """
    return np.exp(-((beta - beta0) ** 2) / (2 * sigma**2))

def gaussian_lambda(t, beta, beta0, sigma, lambda0, omega_drive):
    """
    Return the Gaussian-envelope drive amplitude.
    Params:
        t: Time.
        beta: Inverse temperature.
        beta0: Gaussian center.
        sigma: Gaussian width.
        lambda0: Drive strength.
        omega_drive: Drive frequency.
    Returns:
        Gaussian-envelope drive amplitude at time t.
    """
    return lambda0 * gaussian_envelope(beta, beta0, sigma) * np.cos(omega_drive * t)

def gaussian_lambda_beta_dot(t, beta, beta0, sigma, lambda0, omega_drive):
    """
    Return the beta derivative of the Gaussian drive amplitude.
    Params:
        t: Time.
        beta: Inverse temperature.
        beta0: Gaussian center.
        sigma: Gaussian width.
        lambda0: Drive strength.
        omega_drive: Drive frequency.
    Returns:
        Gaussian drive derivative at time t.
    """
    # Differentiate the Gaussian envelope with respect to beta.
    dG_db = -((beta - beta0) / (sigma**2)) * gaussian_envelope(beta, beta0, sigma)
    return lambda0 * dG_db * np.cos(omega_drive * t)

def H_gauss(t, beta, beta0, sigma, lambda0, omega_drive):
    """
    Build the Hamiltonian for the Gaussian-envelope drive.
    Params:
        t: Time.
        beta: Inverse temperature.
        beta0: Gaussian center.
        sigma: Gaussian width.
        lambda0: Drive strength.
        omega_drive: Drive frequency.
    Returns:
        Driven two-level Hamiltonian.
    """
    # Apply the Gaussian beta envelope to the oscillating drive.
    lam = gaussian_lambda(
        t, beta, beta0, sigma, lambda0=lambda0, omega_drive=omega_drive
    )
    return 0.5*hbar*Omega*sz + lam * sx


###################
####Unitary Evo####
###################
def compute_unitaries(H_fun, beta, timepoints, hbar=1.0, reorthonormalize=True):
    """
    Integrate the Schrodinger equation to compute unitary evolution.
    Params:
        H_fun: Hamiltonian function H(t, beta).
        beta: Inverse temperature.
        timepoints: Times where unitaries are returned.
        hbar: Reduced Planck constant.
        reorthonormalize: Whether to project numerical results back to unitary matrices.
    Returns:
        Array of unitary matrices for each timepoint.
    """
    dim = I2.shape[0]

    #Flatten U for solve_ivp and reshape inside the RHS.
    def ode_rhs(t, U_flat):
        U = U_flat.reshape(dim, dim)
        return (-1j/hbar * H_fun(t, beta) @ U).ravel()

    #Integrate dot{U} = -i H U / hbar.
    sol = solve_ivp(ode_rhs,
                    (timepoints[0], timepoints[-1]),
                    I2.ravel(),
                    t_eval=timepoints,
                    method='RK45', rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError('Unitary integration failed')

    #Reshape solver output into one matrix per timepoint.
    Us = sol.y.T.reshape(-1, dim, dim)
    if reorthonormalize:
        #Remove small numerical drift from unitarity.
        Us = np.array([polar(u)[0] for u in Us])
    return Us


#####################
#####Simulation######
#####################

def Evolve_QFI(beta, timepoints,lambda0, omega_drive, driving_type='lin', d_beta=1e-4,\
               eig_step=10, rng=None, beta_ref=None, sigma_ref=None,\
               compute_cost=False, cost_eps=1e-15):
    """
    Run the QFI simulation for one beta and one drive choice.
    Params:
        beta: Inverse temperature.
        timepoints: Times where QFI values are evaluated.
        driving_type: Drive label ('lin', 'cos', or 'gauss').
        d_beta: Finite-difference step for beta derivatives.
        eig_step: Interval for eigendecomposition-based QFI updates.
        rng: Optional random number generator for Gaussian drive centers.
        beta_ref: Optional fixed Gaussian center.
        sigma_ref: Optional fixed Gaussian width.
        lambda0: Drive strength.
        omega_drive: Drive frequency.
        compute_cost: Whether to compute cost and QFI-to-cost ratio.
        cost_eps: Small offset used to avoid division by zero in the ratio.
    Returns:
        Dictionary with QFI arrays, baseline, optional cost arrays, and drive metadata.
    """
    #initial state
    pi0, d_pi0, L0, F0 = thermal(beta)

    #choose drive
    beta0_used, sigma_used = None, None
    if driving_type == 'lin':
        H_fun = partial(H_lin, lambda0=lambda0)
        lambda_fun = partial(linear_lambda, lambda0=lambda0)
        lambda_dot = partial(linear_lambda_dot, lambda0=lambda0)
    elif driving_type == 'cos':
        H_fun = partial(H_sin, lambda0=lambda0, omega_drive=omega_drive)
        lambda_fun = partial(sinusoidal_lambda, lambda0=lambda0, omega_drive=omega_drive)
        lambda_dot = partial(sinusoidal_lambda_dot, lambda0=lambda0, omega_drive=omega_drive)

    elif driving_type == 'gauss':
        #sigma from equilibrium F0; floor to avoid overflow
        if beta_ref is None: #Choose beta_ref = beta_true
            sigma = 1.0 / np.sqrt(max(F0, 1e-300))
            if rng is None:
                rng = np.random.default_rng()  # in case of truly random choice
            #Sample the Gaussian center near the current beta.
            low  = max(0.0, beta - sigma)
            high = beta + sigma
            beta0 = rng.uniform(low, high)
        else:
            #Reuse a supplied Gaussian envelope.
            beta0 = beta_ref
            sigma = sigma_ref
        H_fun = partial(
            H_gauss, beta0=beta0, sigma=sigma,
            lambda0=lambda0, omega_drive=omega_drive
        )
        lambda_fun = partial(
            gaussian_lambda, beta0=beta0, sigma=sigma,
            lambda0=lambda0, omega_drive=omega_drive
        )
        lambda_dot = partial(
            gaussian_lambda_beta_dot, beta0=beta0, sigma=sigma,
            lambda0=lambda0, omega_drive=omega_drive
        )
        beta0_used, sigma_used = float(beta0), float(sigma)
    else:
        raise ValueError(f'Unknown drive: {driving_type}')

    #Unitaries at beta, beta +- d_beta
    Us   = compute_unitaries(H_fun, beta,timepoints)
    Us_p = compute_unitaries(H_fun, beta + d_beta,timepoints)
    Us_m = compute_unitaries(H_fun, beta - d_beta,timepoints)

    #Approximate beta derivative of Unitary by finite diffs
    Xs_ab = (Us_p - Us_m)/(2 * d_beta)

    nt = len(timepoints)
    full_qfi = np.zeros(nt)
    eig_qfi  = np.zeros(nt)

   
    lambda_dots = np.array([lambda_dot(t, beta) for t in timepoints])
    Xs_loc = np.empty_like(Us)
    for k, U in enumerate(Us):
        #Perturbation in Heisenberg picture
        Vh = U.conj().T @ sx @ U
        deriv = commutator(Vh, pi0)
        Xs_loc[k] = solve_sylvester(pi0, pi0, 2 * deriv)
    Ms = (-1j/hbar) * lambda_dots[:, None, None] * Xs_loc #Integrand of the information current
    integrals = cumtrapz(Ms, timepoints, axis=0, initial=0)

    # compute QFIs
    last_eig = 0.0
    for k, U in enumerate(Us):
        # Evolve the initial thermal state to the current time.
        rho_t = U @ pi0 @ U.conj().T
        # eig QFI every eig_step
        if k % eig_step == 0:
            # Build beta derivative of rho using finite-difference dU/dbeta.
            dU = Xs_ab[k]
            drho = dU @ pi0 @ U.conj().T + U @ d_pi0 @ U.conj().T + U @ pi0 @ dU.conj().T
            lam, V = eigh(rho_t)
            drho_ij = V.conj().T @ drho @ V
            F_eig = 0.0
            # Sum the standard eigenbasis QFI expression.
            for i in range(2):
                for j in range(2):
                    denom = lam[i] + lam[j]
                    if denom > 1e-12:
                        F_eig += (2.0/denom) * abs(drho_ij[i,j])**2
            last_eig = np.real(F_eig)
        eig_qfi[k] = last_eig
        # full QFI from Eq.13
        static_k = U @ L0 @ U.conj().T
        L1_k    = U @ integrals[k] @ U.conj().T
        Ltot    = static_k + L1_k
        full_qfi[k] = np.real(np.trace(rho_t @ Ltot @ Ltot))

    result = {
        'timepoints': timepoints,
        'qfi_full': full_qfi,
        'qfi_eig': eig_qfi,
        'baseline_F0': F0,
        'beta0_used': beta0_used,
        'sigma_used': sigma_used,
        'lambda0': lambda0,
        'omega_drive': omega_drive,
        'cost_computed': bool(compute_cost),
    }

    if compute_cost:
        # Integrate squared drive amplitude and form the QFI-to-cost ratio.
        lambdas = np.array([lambda_fun(t, beta) for t in timepoints])
        cost_t = cumtrapz(lambdas**2, timepoints, initial=0)
        result['cost_t'] = cost_t
        result['qfi_cost_ratio'] = (full_qfi - F0) / (cost_t + cost_eps)

    return result

##############
#####Main#####
##############

if __name__ == '__main__':

    #Simulation parameters
    hbar = 1.0
    Omega = 1.0
    Omega_drive = 2.0
    Lambda0 = 0.01
    Tau = 1/(2*np.pi)
    SEED = 28092 #RNG   
    beta_list = [0.5,0.6,0.7,0.8,0.9,1.0,1.1]
    dt = 0.001 #timestep
    t_final = 6 * np.pi
    timepoints = np.arange(0.0, t_final + dt, dt)
    driving_type = 'gauss'  
    
    cost = True #Flag for the computation of cost ratio

    rng = np.random.default_rng(SEED)  #single seeded generator for all runs
    #This implies subsequent runs will be generated by calls to the same set
    #of seeded beta0 and whatnot

    #Run simulation for each beta and save results
    for beta in beta_list:
        result = Evolve_QFI(
            beta, timepoints,lambda0=Lambda0,\
            omega_drive=Omega_drive, driving_type=driving_type,\
            d_beta=1e-4, eig_step=10, rng=rng,\
            compute_cost=cost
        )

        #prepare result dir
        folder = os.path.join('Results', 'TLS', f'Omega_{Omega}', f'Beta_{beta:.2f}',
                              'Gauss' if driving_type=='gauss' else driving_type.capitalize(),
                              f'{Omega_drive}_{Lambda0}')
        
        os.makedirs(folder, exist_ok=True)
        #save data (+ beta0,sigma if gaussian)
        cost_tag = 'CYes' if cost else 'CNo'
        filename = os.path.join(folder, f'qfi_beta_{beta}_{driving_type}_{cost_tag}.npz')
        save_data = {
            'timepoints': result['timepoints'],
            'qfi_full': result['qfi_full'],
            'qfi_eig': result['qfi_eig'],
            'baseline_F0': result['baseline_F0'],
            'lambda0': result['lambda0'],
            'omega_drive': result['omega_drive'],
            'cost_computed': result['cost_computed'],
        }
        if driving_type == 'gauss':
            save_data['beta0_used'] = result['beta0_used']
            save_data['sigma_used'] = result['sigma_used']
        if cost:
            save_data['cost_t'] = result['cost_t']
            save_data['qfi_cost_ratio'] = result['qfi_cost_ratio']
        np.savez(filename, **save_data)

        beta0 = result['beta0_used']
        sigma = result['sigma_used']
        print(f'Saved QFI data to {filename}' + \
              (f" (beta0={beta0:.6g}, sigma={sigma:.6g})" if beta0 is not None else ""))
