import os

import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import pickle

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
import sys
import json
import random
from math import sin, pi, log, exp, sqrt, floor, ceil, log1p, isnan
from collections import OrderedDict
import numpy

fname = '../native_EDM/data/lorenz_daily.pickle'
with open(fname, 'rb') as f:
    extrema = pickle.load(f).apply(['min', 'max'])


class ExecutionException(Exception):
    def __init__(self, cause, stdout_data, stderr_data):
        self.cause = cause
        self.stdout_data = stdout_data
        self.stderr_data = stderr_data


def run_via_pypy(model_name, params):
    import jsonobject
    import subprocess
    proc = subprocess.Popen(
        ['/usr/bin/env', 'pypy', os.path.join(SCRIPT_DIR, 'models_pypy.py'), model_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False
    )

    if isinstance(params, jsonobject.JSONObject):
        stdin_data = params.dump_to_string()
    else:
        stdin_data = json.dumps(params)
    stdout_data, stderr_data = proc.communicate(stdin_data)
    proc.wait()

    try:
        return jsonobject.load_from_string(stdout_data)
    except Exception as e:
        raise ExecutionException(e, stdout_data, stderr_data)


def sugihara_mirage_correlation(t_max=200, rx=3.8, ry=3.5, Bxy=0.02, Byx=0.1, x0=0.4, y0=0.2):
    '''From Figure 1 in Sugihara et al. Science 2010.'''
    x = numpy.zeros(t_max + 1, dtype=float)
    x[0] = x0
    y = numpy.zeros(t_max + 1, dtype=float)
    y[0] = y0

    for t in range(t_max):
        x[t + 1] = x[t] * (rx - rx * x[t] - Bxy * y[t])
        y[t + 1] = y[t] * (ry - ry * y[t] - Byx * x[t])

    return x, y


def sugihara_example1(
        rng,
        t_max=1000, burnin=200, r1=3.1, D1=3, Sa1=0.4, r2=2.9, D2=3, Sa2=0.35,
        noiseparam=1, psi=0.3
):
    n1series = numpy.zeros(t_max + burnin + 1, dtype=float)
    n2series = numpy.zeros(t_max + burnin + 1, dtype=float)
    redseries = numpy.zeros(t_max + burnin + 1, dtype=float)

    def schaffer(nt, Tt, r, psi):
        return nt * (r * (1.0 - nt)) * numpy.exp(psi * Tt)

    def step_annual(nt0, ntD, TtD, Sa, r, psi):
        return nt0 * Sa + max(schaffer(ntD, TtD, r, psi), 0.0)

    n1series[:max(D1, D2) + 1] = 0.5 * rng.uniform(0, 1, size=max(D1, D2) + 1)
    n2series[:max(D1, D2) + 1] = 0.5 * rng.uniform(0, 1, size=max(D1, D2) + 1)
    whiteseries = -rng.uniform(0, 1, size=redseries.shape) + 0.5

    assert numpy.all(n1series >= 0.0)
    assert numpy.all(n2series >= 0.0)

    for i in range(redseries.shape[0]):
        if i >= noiseparam:
            redseries[i] = numpy.mean(whiteseries[i - noiseparam: i])
        else:
            redseries[i] = 0.0

    pseries = (redseries - numpy.mean(redseries)) / numpy.std(redseries)

    for i in range(max(D1, D2) + 1, n1series.shape[0]):
        n1series[i] = step_annual(n1series[i - 1], n1series[i - 1 - D1], pseries[i - 1 - D1], Sa1, r1, psi)
        n2series[i] = step_annual(n2series[i - 1], n2series[i - 1 - D2], pseries[i - 1 - D2], Sa2, r2, 1.2 * psi)

        if n1series[i] < 0.0 or n2series[i] < 0.0:
            print(n1series[i], n2series[i])

        assert n1series[i] > 0.0
        assert n2series[i] > 0.0

    return n1series[burnin:], n2series[burnin:], pseries[burnin:]


@jit(nopython=True)
def lorenz_step(state, h, ret, n_steps):
    x, y, z = state
    h = h * 1.3744774477447745 / 365 / 2
    for _ in range(n_steps):
        x_dot = 10 * (y - x)
        y_dot = 28 * x - y - x * z
        z_dot = x * y - 2.667 * z
        x, y, z = x + (x_dot * h), y + (y_dot * h), z + (z_dot * h)
    ret[:] = x, y, z


def multistrain_sde(
        random_seed=None,
        dt_euler=None,
        adaptive=None,
        t_end=None,
        dt_output=None,
        n_pathogens=None,
        S_init=None,
        I_init=None,
        mu=None,
        nu=None,
        gamma=None,
        beta0=None,
        beta_change_start=None,
        beta_slope=None,
        psi=None,
        omega=None,
        eps=None,
        sigma=None,

        corr_proc=None,
        sd_proc=None,

        shared_obs=False,
        sd_obs=None,

        shared_obs_C=False,
        sd_obs_C=None,

        tol=None,
        lorenz=None):
    if random_seed is None:
        sys_rand = random.SystemRandom()
        random_seed = sys_rand.randint(0, 2 ** 31 - 1)
    rng = random.Random()
    rng.seed(random_seed)

    pathogen_ids = list(range(n_pathogens))

    stochastic = sum([sd_proc[i] > 0.0 for i in pathogen_ids]) > 0
    assert not (adaptive and stochastic)

    has_obs_error = (sd_obs is not None) and (sum([sd_obs[i] > 0.0 for i in pathogen_ids]) > 0)
    has_obs_error_C = (sd_obs_C is not None) and (sum([sd_obs_C[i] > 0.0 for i in pathogen_ids]) > 0)

    log_mu = log(mu)
    log_gamma = [float('-inf') if gamma[i] == 0.0 else log(gamma[i]) for i in range(n_pathogens)]

    n_output = int(ceil(t_end / dt_output))

    def transformer(x, col):
        scaled = 2 * (x[col] - extrema.iloc[0, col]) / (extrema.iloc[1, col] - extrema.iloc[0, col]) - 1
        assert scaled >= -1
        assert scaled <= 1
        return 1 + eps * scaled

    def beta_t(t,
               pathogen_id,
               weather_state):
        ret = (beta0[pathogen_id] + max(0.0, t - beta_change_start[pathogen_id]) * beta_slope[pathogen_id]) * \
              weather_state[pathogen_id]
        assert ret >= 0
        return ret

    def step(t,
             h,
             logS,
             logI,
             CC,
             weather_state,
             lorenz=None):
        neg_inf = float('-inf')

        sqrt_h = sqrt(h)
        if lorenz is not None:
            assert weather_state.shape == (3,)
            new_weather_state = transformer(weather_state, lorenz)
            assert new_weather_state.shape == (2,)
            assert np.all((new_weather_state >= 0) & (new_weather_state <= 2))
            weather_state = new_weather_state
            # weather_state = [1 + eps[i] * weather_state for i in pathogen_ids]

        log_betas = [log(beta_t(t, i, weather_state=weather_state)) for i in pathogen_ids]
        try:
            logR = [log1p(-(exp(logS[i]) + exp(logI[i]))) for i in pathogen_ids]
        except:
            R = [max(0.0, 1.0 - exp(logS[i]) - exp(logI[i])) for i in pathogen_ids]
            logR = [neg_inf if R[i] == 0 else log(R[i]) for i in pathogen_ids]

        if stochastic:
            if corr_proc == 1.0:
                noise = [rng.gauss(0.0, 1.0)] * n_pathogens
            else:
                noise = [rng.gauss(0.0, 1.0) for i in pathogen_ids]
                if corr_proc > 0.0:
                    assert n_pathogens == 2
                    noise[1] = corr_proc * noise[0] + sqrt(1 - corr_proc * corr_proc) * noise[1]
            for i in pathogen_ids:
                noise[i] *= sd_proc[i]

        dlogS = [0.0 for i in pathogen_ids]
        dlogI = [0.0 for i in pathogen_ids]
        dCC = [0.0 for i in pathogen_ids]

        for i in pathogen_ids:
            dlogS[i] += (exp(log_mu - logS[i]) - mu) * h
            if gamma[i] > 0.0 and logR[i] > neg_inf:
                dlogS[i] += exp(log_gamma[i] + logR[i] - logS[i]) * h
            for j in pathogen_ids:
                if i != j:
                    dlogSRij = sigma[i][j] * exp(log_betas[j] + logI[j])
                    dlogS[i] -= dlogSRij * h
                    if stochastic:
                        dlogS[i] -= dlogSRij * noise[j] * sqrt_h
            dlogS[i] -= exp(log_betas[i] + logI[i]) * h
            dlogI[i] += exp(log_betas[i] + logS[i]) * h
            dCC[i] += exp(log_betas[i] + logS[i] + logI[i]) * h
            if stochastic:
                dlogS[i] -= exp(log_betas[i] + logI[i]) * noise[i] * sqrt_h
                dlogI[i] += exp(log_betas[i] + logS[i]) * noise[i] * sqrt_h
                dCC[i] += exp(log_betas[i] + logS[i] + logI[i]) * noise[i] * sqrt_h
            dlogI[i] -= (nu[i] + mu) * h

        return [logS[i] + dlogS[i] for i in pathogen_ids], \
               [logI[i] + dlogI[i] for i in pathogen_ids], \
               [CC[i] + dCC[i] for i in pathogen_ids]

    def weather_step(t,
                     h,
                     weather_state,
                     lorenz=None):
        if lorenz is None:
            return [
                1 + eps[pathogen_id] * sin(2.0 * pi / psi[pathogen_id] * (t - omega[pathogen_id] * psi[pathogen_id]))
                for pathogen_id in pathogen_ids]
        else:
            ret = np.empty(3, dtype=float)
            lorenz_step(state=weather_state, h=h, ret=ret, n_steps=1)
            return ret

    logS = [log(S_init[i]) for i in pathogen_ids]
    logI = [log(I_init[i]) for i in pathogen_ids]
    if lorenz is None:
        weather = [1, 1]
    else:
        weather = np.empty(3)
        lorenz_step(state=np.random.uniform(low=0.5, high=1.5, size=3),
                    h=1,
                    n_steps=1000,
                    ret=weather)
    CC = [0.0 for i in pathogen_ids]
    h = dt_euler

    ts = [0.0]
    logSs = [logS]
    logIs = [logI]
    CCs = [CC]
    Cs = [CC]
    weathers = [weather]

    if adaptive:
        sum_log_h_dt = 0.0
    for output_iter in range(n_output):
        min_h = h

        t = output_iter * dt_output
        t_next_output = (output_iter + 1) * dt_output

        while t < t_next_output:
            if h < min_h:
                min_h = h

            t_next = t + h
            if t_next > t_next_output:
                t_next = t_next_output
            weather_full = weather_step(t,
                                        t_next - t,
                                        weather,
                                        lorenz=lorenz)
            logS_full, logI_full, CC_full = step(t,
                                                 t_next - t,
                                                 logS,
                                                 logI,
                                                 CC,
                                                 weather_state=weather,
                                                 lorenz=lorenz)
            if adaptive:
                t_half = t + (t_next - t) / 2.0
                weather_half = weather_step(t,
                                            t_half - t,
                                            weather,
                                            lorenz=lorenz)
                logS_half, logI_half, CC_half = step(t,
                                                     t_half - t,
                                                     logS,
                                                     logI,
                                                     CC,
                                                     weather_state=weather,
                                                     lorenz=lorenz)
                weather_half2 = weather_step(t,
                                             t_half - t,
                                             weather_state=weather_half,
                                             lorenz=lorenz)
                logS_half2, logI_half2, CC_half2 = step(t_half,
                                                        t_next - t_half,
                                                        logS_half,
                                                        logI_half,
                                                        CC_half,
                                                        weather_state=weather_half,
                                                        lorenz=lorenz)

                errorS = [logS_half2[i] - logS_full[i] for i in pathogen_ids]
                errorI = [logI_half2[i] - logI_full[i] for i in pathogen_ids]
                errorCC = [CC_half2[i] - CC_full[i] for i in pathogen_ids]
                max_error = max([abs(x) for x in (errorS + errorI + errorCC)])

                if max_error > 0.0:
                    h = 0.9 * (t_next - t) * tol / max_error
                else:
                    h *= 2.0

                if max_error < tol:
                    sum_log_h_dt += (t_next - t) * log(t_next - t)

                    logS = [logS_full[i] + errorS[i] for i in pathogen_ids]
                    logI = [logI_full[i] + errorI[i] for i in pathogen_ids]
                    CC = [CC_full[i] + errorCC[i] for i in pathogen_ids]
                    weather = [weather_full[i] for i in pathogen_ids]
                    t = t_next
            else:
                logS = logS_full
                logI = logI_full
                CC = CC_full
                weather = weather_full
                t = t_next
        ts.append(t)
        logSs.append(logS)
        weathers.append(weather)
        if not has_obs_error:
            logIs.append(logI)
        else:
            if shared_obs:
                obs_err = rng.gauss(0.0, 1.0)
                obs_errs = [obs_err * sd_obs[i] for i in pathogen_ids]
            else:
                obs_errs = [rng.gauss(0.0, sd_obs[i]) for i in pathogen_ids]
            logIs.append([logI[i] + obs_errs[i] for i in pathogen_ids])
        CCs.append(CC)
        if has_obs_error_C:
            if shared_obs_C:
                obs_err = rng.gauss(0.0, 1.0)
                obs_errs = [obs_err * sd_obs_C[i] for i in pathogen_ids]
            else:
                obs_errs = [rng.gauss(0.0, sd_obs_C[i]) for i in pathogen_ids]
        else:
            obs_errs = [0.0 for i in pathogen_ids]
        Cs.append([max(0.0, CCs[-1][i] - CCs[-2][i] + obs_errs[i]) for i in pathogen_ids])

    result = OrderedDict([
        ('t', ts),
        ('logS', logSs),
        ('logI', logIs),
        ('C', Cs),
        ('weather', weathers),
        ('random_seed', random_seed)
    ])
    if adaptive:
        result.dt_euler_harmonic_mean = exp(sum_log_h_dt / t)
    return result


if __name__ == '__main__':
    stdin_data = sys.stdin.read()
    sys.stderr.write(stdin_data)
    params = json.loads(stdin_data)
    result = globals()[sys.argv[1]](**params)
    sys.stderr.write('{0}'.format(result))
    json.dump(result, sys.stdout)
