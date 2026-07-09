us=1e-6
ns=1e-9
MHz=1e6
GHz=1e9
kHz=1e3
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.signal import find_peaks, peak_widths
from windfreak import SynthHD
import time
import numpy as np
import keysight.qcs as qcs

def pca_project_to_1d(X: np.ndarray):
    """
    X: (N,2)
    Returns:
      z: (N,)  PC1 좌표(1D projection)
      pc1: (2,) PC1 unit vector
      mean: (2,) data mean
      X_proj: (N,2) 2D에서 PC1 선으로 투영된 점들(선택)
    """
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (N, 2)")

    pca = PCA(n_components=1)
    z = pca.fit_transform(X).ravel()   # (N,)
    pc1 = pca.components_[0]           # (2,)
    mean = pca.mean_                   # (2,)
    X_proj = mean + np.outer(z, pc1)   # (N,2)
    return z, pc1, mean, X_proj

def fit_gmm_1d(z, n_components=2, random_state=0, n_init=30, reg_covar=1e-10):
    z = np.asarray(z, float).reshape(-1, 1)  # (N,1)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=n_init,
        reg_covar=reg_covar,
        random_state=random_state
    ).fit(z)
    labels = gmm.predict(z)
    probs = gmm.predict_proba(z)
    return gmm, labels, probs

def normal_pdf(x, mu, var):
    return (1.0 / np.sqrt(2*np.pi*var)) * np.exp(-(x-mu)**2 / (2*var))

def plot_gmm_1d(z, gmm, bins=40, density=True, log_y=False, title=""):
    z = np.asarray(z, float).ravel()
    x = np.linspace(z.min(), z.max(), 800)

    weights = gmm.weights_
    means = gmm.means_.ravel()
    vars_ = np.array([c[0, 0] for c in gmm.covariances_])  # (K,)

    plt.figure()
    plt.hist(z, bins=bins, density=density)

    mix = np.zeros_like(x)
    for w, mu, var in zip(weights, means, vars_):
        comp = w * normal_pdf(x, mu, var)
        mix += comp
        plt.plot(x, comp, linewidth=2)

    plt.plot(x, mix, linewidth=2)
    plt.xlabel("z = projection onto PC1")
    plt.ylabel("Density" if density else "Count")
    plt.title(title or f"GMM on PC1 projection (K={gmm.n_components})")
    if log_y:
        plt.yscale("log")
    plt.tight_layout()
    plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks




def manual_classification(iq_data,classification_refs):
    return np.abs(iq_data-classification_refs[0,0])<np.abs(iq_data-classification_refs[0,1])

def prob_g(classified_data):
    return np.sum(classified_data)/len(classified_data)

def prob_e(classified_data):
    return 1-np.sum(classified_data)/len(classified_data)

def prob_rescale(prob_list,min_p,max_p):
    return (np.array(prob_list)-min_p)/(max_p-min_p)



# def pump_init():
#     synth = SynthHD('COM4')
#     synth.init()

#     return synth

# def pump_on(synth):
#     synth[0].enable = False
#     synth[1].enable = True

#     synth[1].power = float(7)
#     synth[1].frequency = float(7.21*GHz)

#     time.sleep(0.5)

# def pump_off(synth):
#     synth[1].enable = False
#     synth.close()
import time
from windfreak import SynthHD
def pump_init(max_wait_minutes=10, retry_interval=60):


    start = time.time()
    attempt = 0

    while True:
        attempt += 1
        try:
            synth = SynthHD('socket://163.152.38.107:4000')
            synth.init()
            if attempt > 1:
                print(f"Windfreak 연결 성공! ({attempt}번째 시도)")
            return synth
        except Exception as e:
            msg = str(e)
            is_busy = 'BUSY' in msg or 'in use' in msg

            print(f"[debug] attempt {attempt}: {type(e).__name__}: {e}")

            # BUSY가 아니면 재시도해도 소용없으니 바로 올림
            if not is_busy:
                raise RuntimeError(
                    f"Windfreak 연결 실패 (사용 중 아님). "
                    f"브릿지/IP/장비 상태 확인 필요: {e}"
                )

            elapsed_min = (time.time() - start) / 60
            if elapsed_min >= max_wait_minutes:
                raise RuntimeError(
                    f"Windfreak가 {max_wait_minutes}분 동안 계속 사용 중입니다. "
                    f"(마지막 에러: {e})"
                )

            print(
                f"⏳ Windfreak 사용 중입니다. {retry_interval}초 후 재시도합니다. "
                f"(경과: {elapsed_min:.1f}분 / 최대 {max_wait_minutes}분)"
            )
            time.sleep(retry_interval)
def pump_on(synth):
    synth[0].enable = False
    synth[1].enable = True

    synth[0].power = float(7)
    synth[0].frequency = float(7.21*GHz)

    time.sleep(0.5)

def pump_off(synth):
    synth[1].enable = False
    synth.close()



def force_label0_is_g(gmm, labels, probs, n0):
    """
    gmm: sklearn GaussianMixture (K=2)
    labels: (N,) from gmm.predict
    probs: (N,2) from gmm.predict_proba
    n0: int, 앞쪽 g-prepare 샷 개수 (labels0 길이)

    반환:
      gmm2, labels2, probs2  (0 ≡ g, 1 ≡ e 로 정렬)
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    labels0 = labels[:n0]
    # g-prep에서 더 많이 나온 컴포넌트를 g로 정의
    # (동률이면 그대로 두지만, 현실에선 거의 안 나옴)
    cnt0 = np.sum(labels0 == 0)
    cnt1 = np.sum(labels0 == 1)

    if cnt0 >= cnt1:
        # 이미 0이 g쪽으로 더 많음 → 그대로
        return gmm, labels, probs
    else:
        # 0/1 swap
        labels_swapped = 1 - labels
        probs_swapped = probs[:, [1, 0]]

        # gmm 파라미터도 swap (원하면 안 해도 되지만, 일관성 위해 추천)
        gmm.weights_ = gmm.weights_[[1, 0]]
        gmm.means_   = gmm.means_[[1, 0]]
        gmm.covariances_ = gmm.covariances_[[1, 0]]
        # precision 쪽도 있을 수 있음 (sklearn이 내부에서 씀)
        if hasattr(gmm, "precisions_"):
            gmm.precisions_ = gmm.precisions_[[1, 0]]
        if hasattr(gmm, "precisions_cholesky_"):
            gmm.precisions_cholesky_ = gmm.precisions_cholesky_[[1, 0]]

        return gmm, labels_swapped, probs_swapped

def print_channel_summary(
    mapper,
    qubit_channels="xy_pulse",
    ro_drive_channels="readout_channels",
    ro_acquire_channels="acquire_channels",
):
    """큐빗 XY / 리드아웃 드라이브 / 리드아웃 acquire 채널의
    물리 주소와 LO 주파수를 출력."""

    def addr_tuple(pc):
        a = pc.address
        return (a.chassis, a.slot, a.channel)

    def lo_str(pc):
        s = pc.settings
        if "lo_frequency" in s.setting_names:
            v = s.lo_frequency.value
        else:
            # 디지타이저는 자체 LO가 없으므로 상단 다운컨버터에서 읽음
            dnc = mapper.get_downconverter(pc.address)
            v = dnc.settings.lo_frequency.value if dnc is not None else None
        return f"{v / 1e9:g} GHz" if v is not None else "미설정"

    by_name = {getattr(ch, "name", None): ch for ch in mapper.channels}
    parts = []

    qch = by_name.get(qubit_channels)
    if qch is not None:
        pcs = mapper.get_physical_channels(qch)
        labels = list(getattr(qch, "labels", range(len(pcs))))
        for label, pc in zip(labels, pcs):
            parts.append(f"q{label}: {addr_tuple(pc)}, LO {lo_str(pc)}")

    rch = by_name.get(ro_drive_channels)
    if rch is not None:
        pc = mapper.get_physical_channels(rch)[0]
        parts.append(f"ro_drive: {addr_tuple(pc)}, LO {lo_str(pc)}")

    ach = by_name.get(ro_acquire_channels)
    if ach is not None:
        pc = mapper.get_physical_channels(ach)[0]
        parts.append(f"ro: {addr_tuple(pc)}, LO {lo_str(pc)}")

    if not parts:
        print(f"(채널을 못 찾았습니다 — 컬렉션 이름 확인: {list(by_name)})")
        return

    print("\n".join(parts))

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

