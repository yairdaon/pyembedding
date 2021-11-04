import numpy as np
from matplotlib import pyplot as plt

from models import multistrain_sde


def main():
    result = multistrain_sde(dt_euler=1,
                             adaptive=False,
                             t_end=20 * 365,
                             dt_output=7,
                             n_pathogens=2,
                             S_init=[0.9, 0.96],
                             I_init=[0.001, 0.002],
                             
                             mu=1 / 30 / 365,
                             nu=0.2 * np.ones(2),
                             gamma=np.zeros(2),
                             beta0=np.array([0.3, 0.25]),
                             beta_change_start=np.zeros(2),
                             beta_slope=np.zeros(2),
                             psi=np.ones(2) * 365,
                             omega=np.zeros(2),
                             eps=[1, 0.1],  # 0.1 * np.ones(2),
                             sigma=np.array([[1, 0], [0.2, 1]]),
                             corr_proc=1,
                             sd_proc=np.ones(2) * 0.0,
                             shared_obs=False,
                             sd_obs=np.ones(2) * 0.0,
                             shared_obs_C=False,
                             sd_obs_C=np.array([0.1, 0.2]) * 0.0,
                             tol=1e-3)
    logS = np.vstack(result['logS'])
    logI = np.vstack(result['logI'])
    weather = np.vstack(result['weather'])
    T = result['t']

    fig, axes = plt.subplots(nrows=3, sharex=True)
    ax = axes[0]
    ax.plot(T, logS[:, 1], label='logS1')
    ax.plot(T, logS[:, 0], label='logS0')
    ax.legend()

    ax = axes[1]
    ax.plot(T, logI[:, 0], label='logI0')
    ax.plot(T, logI[:, 1], label='logI1')
    ax.legend()

    ax = axes[2]
    ax.plot(T, weather[:, 1], label='weather1')
    ax.plot(T, weather[:, 0], label='weather0')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
