us=1e-6
ns=1e-9
MHz=1e6
GHz=1e9
kHz=1e3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths,savgol_filter
from scipy.optimize import curve_fit

def find_dips(freq, signal, smooth=True):
    y = np.asarray(signal)

    # 필요하면 smoothing
    if smooth:
        # window_length는 홀수여야 함
        window_length = min(41, len(y) // 2 * 2 - 1)
        if window_length >= 5:
            y_smooth = savgol_filter(y, window_length=window_length, polyorder=3)
        else:
            y_smooth = y
    else:
        y_smooth = y

    # dip을 찾기 위해 -y_smooth에서 peak 탐색
    dip_indices, properties = find_peaks(
        -y_smooth
    )

    dip_freqs = freq[dip_indices]

    return dip_freqs


def t1_model(t, T1, A, C):
    return A *(1-np.exp(-t/T1)) + C

def t1_fit(t_list,signals,t1_init=100*us):
    T1_init = t1_init
    A_init = signals[-1]-signals[0]
    C_init = signals[0]-A_init

    popt, _ = curve_fit(t1_model,t_list,signals,p0=[T1_init,A_init,C_init])

    T1_fit, A_fit, C_fit = popt

    plt.figure(figsize=(8,4))
    plt.plot(t_list/us,signals)
    plt.plot(t_list/us,[t1_model(t,T1_fit,A_fit,C_fit) for t in t_list])
    plt.xlabel("Time ($\mu$s)")
    plt.ylabel("Population")

    plt.text(0.7,0.8,
    rf"$T_1={T1_fit/us:.3f}\,\mu s$",
    size=13, 
    transform=plt.gca().transAxes,
    verticalalignment='top',
    bbox=dict(boxstyle='round',facecolor='white',alpha=0.8)
    )
    plt.show()
    return T1_fit



def ramsey_model(t, T2, delta, phase, A, C):
    return A *(np.exp(-t/T2))*np.cos(2*np.pi*delta*t+phase)+C

def t2_fit(t_list,signals,T2_guess,delta_guess):
    T2_init = T2_guess
    delta_init=delta_guess
    phase_init=0
    A_init = signals[-1]-signals[0]
    C_init = signals[0]

    popt, _ = curve_fit(ramsey_model,t_list,signals,p0=[T2_init,delta_init,phase_init,A_init,C_init],maxfev=10000)

    T2_fit, delta_fit, phase_fit, A_fit,C_fit = popt


    plt.figure(figsize=(8,4))
    plt.plot(t_list/us,signals)
    plt.plot(t_list/us,[ramsey_model(t,T2_fit, delta_fit, phase_fit, A_fit,C_fit) for t in t_list])
    plt.xlabel("Time ($\mu$s)")
    plt.ylabel("Population")


    plt.text(0.7,0.9,
    rf"$T_2={T2_fit/us:.3f}\,\mu s$",
    size=13, 
    transform=plt.gca().transAxes,
    verticalalignment='top',
    bbox=dict(boxstyle='round',facecolor='white',alpha=0.8)
    )

    plt.text(0.7,0.8,
    rf"$\Delta={delta_fit*1e-6:.3f}\,MHz$",
    size=13, 
    transform=plt.gca().transAxes,
    verticalalignment='top',
    bbox=dict(boxstyle='round',facecolor='white',alpha=0.8)
    )
    return T2_fit, delta_fit



def lorentzian(x, y0, A, f0, kappa):
    """Lorentzian in form y = y0 + A * ( (kappa/2)^2 / ((x-f0)^2 + (kappa/2)^2) )
       Here kappa is FWHM (Hz)."""
    return y0 + A * ( (0.5*kappa)**2 / ((x - f0)**2 + (0.5*kappa)**2) )

def fit_lorentzian(x_data, y_data, plot=True):
    x = np.array(x_data)
    y = np.array(y_data)

    # initial guesses
    y0_guess = np.median(np.r_[y[:5], y[-5:]])  # baseline ~ edges
    # decide peak or dip by comparing median to max/min
    if np.max(y) - y0_guess > y0_guess - np.min(y):
        # peak-like
        A_guess = np.max(y) - y0_guess
        f0_guess = x[np.argmax(y)]
    else:
        # dip-like (negative amplitude)
        A_guess = np.min(y) - y0_guess
        f0_guess = x[np.argmin(y)]
    kappa_guess = (np.max(x) - np.min(x)) / 20.0  # rough guess: 1/20 of span

    p0 = [y0_guess, A_guess, f0_guess, kappa_guess]

    # bounds: allow kappa positive
    lower = [-np.inf, -np.inf, np.min(x), 1e-12]
    upper = [ np.inf,  np.inf, np.max(x), np.max(x)-np.min(x)]

    try:
        popt, pcov = curve_fit(lorentzian, x, y, p0=p0, bounds=(lower,upper), maxfev=5000)
    except Exception as e:
        raise RuntimeError(f"Fitting failed: {e}")

    y0, A, f0, kappa = popt
    perr = np.sqrt(np.diag(pcov))  # 1-sigma uncertainties
    y0_err, A_err, f0_err, kappa_err = perr

    Q = f0 / kappa if kappa != 0 else np.nan
    # propagate uncertainty for Q = f0 / kappa (assuming covariances small):
    Q_err = Q * np.sqrt((f0_err / f0)**2 + (kappa_err / kappa)**2) if (f0>0 and kappa>0) else np.nan

    # print results
    print("Fit results:")
    print(f"  f0     = {f0:.6e} ± {f0_err:.2e}  [Hz]")
    print(f"  kappa  = {kappa:.6e} ± {kappa_err:.2e}  [Hz]  (FWHM)")
    print(f"  Q      = {Q:.6e} ± {Q_err:.2e}")

    if plot:
        x_fit = np.linspace(np.min(x), np.max(x), 2000)
        y_fit = lorentzian(x_fit, *popt)

        plt.figure(figsize=(6,4))
        plt.plot(x, y, 'o', ms=4, label='data')
        plt.plot(x_fit, y_fit, '-', label='lorentzian fit')
        plt.axvline(f0, color='k', linestyle='--', lw=0.8, label=f'f0={f0:.6e} Hz')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Signal (linear)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    out = {
        'popt': popt,
        'pcov': pcov,
        'y0': y0,
        'A': A,
        'f0': f0,
        'kappa': kappa,
        'y0_err': y0_err,
        'A_err': A_err,
        'f0_err': f0_err,
        'kappa_err': kappa_err,
        'Q': Q,
        'Q_err': Q_err
    }
    return out



def amplitude_rabi_model(amp,Omega,phi,A,C):
    return A*np.cos(2*np.pi*Omega*amp+phi)+C

def amplitude_rabi_fit(amp_list, signals):
    smoothed = savgol_filter(signals, window_length=15, polyorder=3)
    centered = smoothed - np.mean(smoothed)

    n_crossings = np.sum(np.diff(np.sign(centered)) != 0)
    Omega_init = 0.5 * n_crossings / (amp_list[-1] - amp_list[0])
    A_init = (np.max(signals) - np.min(signals)) / 2
    C_init = np.average(signals)
    phi_init = 0
    popt, _ = curve_fit(
        amplitude_rabi_model,
        amp_list,
        signals,
        p0=[Omega_init, phi_init, A_init, C_init],
        bounds=(
            [0, -np.pi, -np.inf, -np.inf],
            [np.inf, np.pi, np.inf, np.inf]
        )
    )
    Omega_fit, phi_fit, A_fit, C_fit = popt

    # --- amp_list 범위 내 첫 crest/trough 찾기 ---
    amp_min, amp_max = amp_list[0], amp_list[-1]
    # 범위 안에 들어올 수 있는 k 후보를 충분히 넓게 잡음
    k_span = int(np.ceil((amp_max - amp_min) * Omega_fit)) + 2
    ks = np.arange(-k_span, k_span + 1)

    # cos(2π·Ω·amp + φ) + C 모델 기준
    crests = (ks - phi_fit / (2 * np.pi)) / Omega_fit # 2π·Ω·amp + φ = 2π·k
    troughs = (ks + 0.5 - phi_fit / (2 * np.pi)) / Omega_fit # 2π·Ω·amp + φ = π + 2π·k

    crests_in = np.sort(crests[(crests >= amp_min) & (crests <= amp_max)])
    troughs_in = np.sort(troughs[(troughs >= amp_min) & (troughs <= amp_max)])

    Phi_amp2 = crests_in[0] if crests_in.size else np.nan # 1st crest in range
    Phi_amp1 = troughs_in[0] if troughs_in.size else np.nan # 1st trough in range

    plt.figure(figsize=(8, 4))
    plt.plot(amp_list, signals)
    plt.plot(amp_list, [amplitude_rabi_model(amp, Omega_fit, phi_fit, A_fit, C_fit) for amp in amp_list])
    if not np.isnan(Phi_amp1):
        plt.axvline(x=Phi_amp1, color='red', linestyle='--')
    if not np.isnan(Phi_amp2):
        plt.axvline(x=Phi_amp2, color='Green', linestyle='--')

    print(f" 1st crest = {Phi_amp2:.4f} ")
    print(f" 1st trough = {Phi_amp1:.4f} ")
    return Phi_amp2, Phi_amp1



def gaussian(x,A,x0,s,bl): return A*np.exp(-(x-x0)**2/(2*s**2)) + bl


def find_out_and_back(
        amp_scan_values,
        phase_values,
        Z,
        prominence=0.07,
        init_range=10,
        discard=16):

    # Ensure inputs are 1D arrays
    x = np.asarray(amp_scan_values).flatten()
    y = np.asarray(phase_values).flatten() * 180 / np.pi - 180  # convert rad to deg and shift

    steps = len(x)
    center_xs = np.zeros(steps)

    for j in range(steps):

        try:
            xn = y
            yn = Z[:, j]  # Use data as is for bright peaks

            peaks, _ = find_peaks(yn, prominence=prominence)

            if len(peaks) == 0:
                center_xs[j] = np.nan
                continue

            if j == 0:
                i = peaks[np.argmax(yn[peaks])]
            else:
                i = peaks[np.argmin(np.abs(xn[peaks] - center_xs[j-1]))]

            L = max(0, i - init_range)
            R = min(len(xn), i + init_range)

            xm = xn[L:R]
            ym = yn[L:R]

            if len(xm) < 5:
                center_xs[j] = np.nan
                continue

            p0 = [ym.max() - ym.min(), xn[i], 10, ym.min()]

            popt, _ = curve_fit(gaussian, xm, ym, p0=p0)

            center_xs[j] = popt[1]

        except Exception as e:
            print(f"Error at column {j}: {e}")
            center_xs[j] = np.nan

    # Discard unreliable points at high amplitudes
    center_xs[discard:] = np.nan

    plt.figure()
    plt.imshow(
        Z,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        aspect="auto"
    )
    plt.plot(x, center_xs, 'ro', label="ridge")
    plt.xlabel("amplitude (arb)")
    plt.ylabel("angle (deg)")
    plt.legend()
    plt.show()

    return center_xs




"""
Hahn echo (spin echo) 실험 데이터에서 T2를 추출하는 피팅/플롯 함수.

핵심 아이디어
  1) 복소 IQ 데이터를 신호 변화가 가장 큰 축(주성분)으로 투영해 1D 신호로 만든다.
     - readout 위상에 무관하게 동적 범위(신호의 변화)를 한 축에 모은다.
  2) 감쇠 모델  y(t) = A * exp(-(t/T2)^p) + C  로 비선형 최소제곱 피팅한다.
     - p=1: 지수 감쇠(백색 노이즈에 가까운 환경)
     - p=2: 가우시안 감쇠(1/f 류 저주파 노이즈가 지배적일 때)
     - exponent=None: p를 자유 파라미터로 두어 데이터가 결정하게 함(stretched exp).

주의: tlist는 보통 '총 자유 진화 시간'(= 2*tau). 실험 시퀀스에서 x축을 무엇으로
잡았는지에 따라 T2 정의가 달라지므로 입력 tlist 정의를 확인할 것.
"""

def project_iq(iq):
    """복소 IQ 배열을 주성분(최대 분산) 축으로 투영해 1D 실수 신호로 변환.

    Parameters
    ----------
    iq : array_like (complex)
        측정 IQ 데이터. I + 1j*Q 형태의 복소 배열.

    Returns
    -------
    proj : ndarray
        주성분 축으로 투영된 1D 신호. 감쇠가 위->아래로 보이도록 부호를 맞춤.
    """
    iq = np.asarray(iq)
    I = np.real(iq)
    Q = np.imag(iq)

    data = np.vstack([I - I.mean(), Q - Q.mean()])
    cov = np.cov(data)
    eigvals, eigvecs = np.linalg.eigh(cov)      # 오름차순 고유값
    principal = eigvecs[:, -1]                   # 최대 분산 방향
    proj = principal @ data

    # 초반부 평균이 후반부보다 작으면 부호 뒤집어 '감쇠' 형태로 통일
    n = max(1, len(proj) // 10)
    if proj[:n].mean() < proj[-n:].mean():
        proj = -proj
    return proj

def _decay(t, A, T2, p, C):
    return A * np.exp(-(t / T2) ** p) + C

def fit_hahn_echo_t2(tlist, iq, exponent=None):
    """Hahn echo 데이터에서 T2를 피팅.

    Parameters
    ----------
    tlist : array_like
        자유 진화 시간 배열 (초 단위 권장).
    iq : array_like (complex)
        측정된 IQ 데이터.
    exponent : float or None, optional
        감쇠 지수 p를 고정할 값. None이면 p도 자유 파라미터로 피팅(기본).
        지수 감쇠를 강제하려면 exponent=1, 가우시안이면 exponent=2.

    Returns
    -------
    result : dict
        T2, T2_err, A, p, C, popt(전체 파라미터), 그리고 투영 신호 t/y 포함.
    """
    t = np.asarray(tlist, dtype=float)
    y = project_iq(iq)

    # 초기값 추정
    n = max(1, len(y) // 10)
    C0 = y[-n:].mean()
    A0 = y[0] - C0
    target = C0 + A0 / np.e                       # 1/e 지점으로 T2 대략 추정
    idx = np.argmin(np.abs(y - target))
    T20 = max(t[idx], (t[-1] - t[0]) / 5)

    if exponent is None:
        p0 = [A0, T20, 1.0, C0]
        bounds = ([-np.inf, 1e-12, 0.3, -np.inf],
                  [np.inf, np.inf, 4.0, np.inf])
        popt, pcov = curve_fit(_decay, t, y, p0=p0, bounds=bounds, maxfev=20000)
    else:
        f = lambda tt, A, T2, C: _decay(tt, A, T2, exponent, C)
        popt3, pcov3 = curve_fit(f, t, y, p0=[A0, T20, C0], maxfev=20000)
        popt = np.array([popt3[0], popt3[1], exponent, popt3[2]])
        pcov = np.zeros((4, 4))
        pcov[np.ix_([0, 1, 3], [0, 1, 3])] = pcov3

    perr = np.sqrt(np.diag(pcov))
    return {
        "T2": popt[1],
        "T2_err": perr[1],
        "A": popt[0],
        "p": popt[2],
        "C": popt[3],
        "popt": popt,
        "t": t,
        "y": y,
    }

def plot_hahn_echo_t2(tlist, iq, exponent=None, time_unit="us", ax=None):
    """Hahn echo 데이터를 피팅하고 결과를 플롯.

    Parameters
    ----------
    tlist, iq, exponent : fit_hahn_echo_t2 참고.
    time_unit : {"s", "ms", "us", "ns"}
        x축 및 T2 표기에 쓸 시간 단위.
    ax : matplotlib.axes.Axes, optional
        그릴 축. None이면 새로 생성.

    Returns
    -------
    result : dict
        fit_hahn_echo_t2의 반환값.
    """


    scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[time_unit]
    res = fit_hahn_echo_t2(tlist, iq, exponent=exponent)
    t, y, popt = res["t"], res["y"], res["popt"]

    t_fit = np.linspace(t.min(), t.max(), 500)
    y_fit = _decay(t_fit, *popt)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(t * scale, y, "o", ms=4, alpha=0.6, label="data (projected)")
    ax.plot(t_fit * scale, y_fit, "-", lw=2, color="C3", label="fit")

    label = (f"$T_2$ = {res['T2'] * scale:.3g} ± {res['T2_err'] * scale:.2g} {time_unit}\n"
             f"p = {res['p']:.2f}")
    ax.text(0.55, 0.85, label, transform=ax.transAxes,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax.set_xlabel(f"Free evolution time ({time_unit})")
    ax.set_ylabel("Signal (projected, a.u.)")
    ax.set_title("Hahn Echo $T_2$ Fit")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return res



################# normalized_variance 계산 추가 #######################3333

def calculate_normalized_variance(y_data, y_fit, eps=1e-12, ddof=0):
    """
    data와 fitting curve 사이의 normalized residual variance 계산.

    normalized_var = var(y_data - y_fit) / var(y_data)

    값 해석:
        0에 가까움  : fitting이 data를 잘 설명함
        1에 가까움  : fitting이 평균값 모델 정도밖에 설명 못함
        1보다 큼     : fitting이 data 평균보다도 못함

    Parameters
    ----------
    y_data : array-like
        실제 측정 데이터.
    y_fit : array-like
        fitting model이 같은 x point에서 예측한 값.
    eps : float
        y_data variance가 0에 가까울 때 division 방지용.
    ddof : int
        variance 계산 자유도. 기본 0.

    Returns
    -------
    normalized_var : float
    """

    y_data = np.asarray(y_data, dtype=float).squeeze().reshape(-1)
    y_fit = np.asarray(y_fit, dtype=float).squeeze().reshape(-1)

    if y_data.shape != y_fit.shape:
        raise ValueError(
            "y_data and y_fit must have same shape. "
            f"got y_data.shape={y_data.shape}, y_fit.shape={y_fit.shape}"
        )

    finite_mask = np.isfinite(y_data) & np.isfinite(y_fit)

    if np.count_nonzero(finite_mask) < 2:
        return np.nan

    y = y_data[finite_mask]
    yf = y_fit[finite_mask]

    residual = y - yf

    residual_var = float(np.var(residual, ddof=ddof))
    data_var = float(np.var(y, ddof=ddof))

    if data_var < eps:
        return np.nan

    return residual_var / data_var

def fit_lorentzian_with_quality(x_data, y_data, plot=False):
    """
    fit_lorentzian()을 실행하고,
    같은 x_data point에서 y_fit을 계산한 뒤 normalized_var를 붙인다.
    """

    fit_result = fit_lorentzian(
        x_data,
        y_data,
        plot=plot,
    )

    y_fit = lorentzian(
        np.asarray(x_data, dtype=float),
        *fit_result["popt"],
    )

    fit_result["fit_x"] = np.asarray(x_data, dtype=float)
    fit_result["fit_y"] = y_fit
    fit_result["y"] = np.asarray(y_data, dtype=float)
    fit_result["normalized_var"] = calculate_normalized_variance(
        y_data,
        y_fit,
    )

    return fit_result

def fit_amplitude_rabi_with_quality(amp_list, signals, plot=False):
    """
    amplitude Rabi fitting을 수행하고,
    x90/x180 후보와 normalized_var를 함께 반환.
    """

    amp_list = np.asarray(amp_list, dtype=float).squeeze().reshape(-1)
    signals = np.asarray(signals, dtype=float).squeeze().reshape(-1)

    if len(amp_list) != len(signals):
        raise RuntimeError(
            f"Rabi length mismatch: len(amp_list)={len(amp_list)}, "
            f"len(signals)={len(signals)}"
        )

    if len(amp_list) < 5:
        raise RuntimeError("Too few points for Rabi fitting.")

    window_length = min(15, len(signals) // 2 * 2 - 1)

    if window_length >= 5:
        smoothed = savgol_filter(
            signals,
            window_length=window_length,
            polyorder=3,
        )
    else:
        smoothed = signals.copy()

    centered = smoothed - np.mean(smoothed)

    n_crossings = np.sum(np.diff(np.sign(centered)) != 0)
    span = amp_list[-1] - amp_list[0]

    if span <= 0:
        raise RuntimeError("Invalid amplitude sweep axis.")

    Omega_init = max(0.5 * n_crossings / span, 1e-9)
    A_init = (np.max(signals) - np.min(signals)) / 2
    C_init = np.average(signals)
    phi_init = 0.0

    popt, pcov = curve_fit(
        amplitude_rabi_model,
        amp_list,
        signals,
        p0=[Omega_init, phi_init, A_init, C_init],
        bounds=(
            [0, -np.pi, -np.inf, -np.inf],
            [np.inf, np.pi, np.inf, np.inf],
        ),
        maxfev=10000,
    )

    Omega_fit, phi_fit, A_fit, C_fit = popt

    amp_min, amp_max = amp_list[0], amp_list[-1]
    k_span = int(np.ceil((amp_max - amp_min) * Omega_fit)) + 2
    ks = np.arange(-k_span, k_span + 1)

    crests = (ks - phi_fit / (2 * np.pi)) / Omega_fit
    troughs = (ks + 0.5 - phi_fit / (2 * np.pi)) / Omega_fit

    crests_in = np.sort(
        crests[(crests >= amp_min) & (crests <= amp_max)]
    )
    troughs_in = np.sort(
        troughs[(troughs >= amp_min) & (troughs <= amp_max)]
    )

    first_crest = crests_in[0] if crests_in.size else np.nan
    first_trough = troughs_in[0] if troughs_in.size else np.nan

    y_fit = amplitude_rabi_model(
        amp_list,
        *popt,
    )

    normalized_var = calculate_normalized_variance(
        signals,
        y_fit,
    )

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(amp_list, signals, "o", ms=4, label="data")
        plt.plot(amp_list, y_fit, "-", label="Rabi fit")

        if np.isfinite(first_trough):
            plt.axvline(
                x=first_trough,
                color="red",
                linestyle="--",
                label="first trough",
            )

        if np.isfinite(first_crest):
            plt.axvline(
                x=first_crest,
                color="green",
                linestyle="--",
                label="first crest",
            )

        plt.xlabel("Amplitude")
        plt.ylabel("Signal")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "status": "ok",
        "first_crest": float(first_crest),
        "first_trough": float(first_trough),
        "popt": popt,
        "pcov": pcov,
        "fit_x": amp_list,
        "fit_y": y_fit,
        "y": signals,
        "normalized_var": normalized_var,
    }

def fit_t1_with_quality(t_list, signals, t1_init=100 * us, plot=False):
    """
    T1 fitting을 수행하고 normalized_var를 함께 반환.
    """

    t_list = np.asarray(t_list, dtype=float).squeeze().reshape(-1)
    signals = np.asarray(signals, dtype=float).squeeze().reshape(-1)

    if len(t_list) != len(signals):
        raise RuntimeError(
            f"T1 length mismatch: len(t_list)={len(t_list)}, "
            f"len(signals)={len(signals)}"
        )

    T1_init = float(t1_init)
    A_init = signals[-1] - signals[0]
    C_init = signals[0] - A_init

    popt, pcov = curve_fit(
        t1_model,
        t_list,
        signals,
        p0=[T1_init, A_init, C_init],
        maxfev=10000,
    )

    T1_fit, A_fit, C_fit = popt

    y_fit = t1_model(
        t_list,
        *popt,
    )

    normalized_var = calculate_normalized_variance(
        signals,
        y_fit,
    )

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(t_list / us, signals, "o", ms=4, label="data")
        plt.plot(t_list / us, y_fit, "-", label="T1 fit")
        plt.xlabel("Time ($\\mu$s)")
        plt.ylabel("Signal")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "status": "ok",
        "T1": float(T1_fit),
        "A": float(A_fit),
        "C": float(C_fit),
        "popt": popt,
        "pcov": pcov,
        "fit_x": t_list,
        "fit_y": y_fit,
        "y": signals,
        "normalized_var": normalized_var,
    }

def fit_hahn_echo_t2_with_quality(t_list, iq_data, exponent=None):
    """
    Hahn echo T2 fitting을 수행하고 normalized_var를 함께 반환.
    """

    fit_result = fit_hahn_echo_t2(
        t_list,
        iq_data,
        exponent=exponent,
    )

    t = np.asarray(fit_result["t"], dtype=float).squeeze().reshape(-1)
    y = np.asarray(fit_result["y"], dtype=float).squeeze().reshape(-1)
    popt = fit_result["popt"]

    y_fit = _decay(
        t,
        *popt,
    )

    fit_result["fit_x"] = t
    fit_result["fit_y"] = y_fit
    fit_result["normalized_var"] = calculate_normalized_variance(
        y,
        y_fit,
    )

    return fit_result
