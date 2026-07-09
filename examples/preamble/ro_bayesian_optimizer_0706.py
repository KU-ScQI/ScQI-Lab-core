import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import keysight.qcs as qcs
import html

try:
    from IPython.display import display, HTML
except Exception:
    display = None
    HTML = None

from skopt import gp_minimize
from skopt.space import Real
from skopt.sampler import Sobol

from keysight.qcs.experiments import ResonatorSpectroscopy, ResonatorSpectroscopy2D
from keysight.qcs.experiments import QubitSpectroscopy, RabiExperiment, T1Experiment, RamseyExperiment, IQDistributionExperiment

from preamble.pump_utils import *
from preamble.pca_gmm_utils import *
from preamble.fitter import *
from preamble.calibration import *
from preamble.classifications import *

"""
logic 설명:
해당 코드는 bayesian optimizer를 기반으로 한 readout paramter를 자동으로 잡는다.

베이지안 최적화:
베이지안 최적화는 parameter와 fidelity의 관계를 추정하는 surrogate model과 다음 탐색 paramter를 정하는 acqusition function이 존재한다.
탐색을 통해 surrogate model를 점점 업데이트 해나가며 acqusition fuction이 더 나은 fidelity를 얻을 수 있는 적절한 parameter를 선택해서 탐색하길 기대한다.

이때 acqusition function은 exploration과 exploitation을 적절하게 이용해야 한다. 만약 충분한 탐색이 이루어지지 않은채로 현재까지의 정보를 바탕으로
빠르게 parameter로 수렴하거나 학습을 끝낸다면 local optimum에 빠질 수 있다. 

따라서 해당 코드에서는 5가지 개선점을 제시한다. 
1. 초기 range를 충분히 크게 잡은 후 low-shot으로 128개의 sobol점을 잡아 대략적인 surrogate모델을 제공한다.
2. acqusition function을 EI가 아닌 LCB를 사용한다. EI는 개선이 기대되는 점만을 잡지만 LCB는 기대 fidelity에 불확실도에 kappa를 곱해 더하여
고려하기에 탐색하지 않은 영역을 탐색할 수 있도록 돕는다. kappa는 초기에 크게 잡고 점점 줄여 처음에는 탐색을, 나중에는 수렴을 기대한다.
3. multi-start L-BFGS를 지원한다. 1개의 기대하는 최적의 acqusition 점이 아닌, 10개의 점을 뽑아서 optimizer에 돌려 그중에서 최고값을 가지고 측정한다.  
4. 종료시점을 더욱 엄격하게 잡아, 확실하게 학습이 끝냈을때 종료되도록 하였다. parameter의 range가 일정이상 내려갔을 때&모든 parameter가 zoom-in이면서 boundary에 
best값이 위치하지 않을 시, fidelity개선이 정체되었을 시 학습이 충분히 이루어졌을 것으로 판단, 종료되도록 하였다. 다만 cycle이 모두 소진되면 자동으로 종료한다.
더불어 fidelity개선이 정체되었는데, range가 크거나 boundary에 best값이 있으면 range를 다시 늘려 탐색한다.
5. center를 1개의 best값이 아니라 top_k로 cycle에서 뽑은 높은 fidelity순으로 k의 parameter의 평균을 구해, 해당 평균값으로 shift, zoom-in등을 robust하게 판단한다. 

이를 통해 기존 코드에서 예상되는 local-optimum문제와 이른 탐색 종료의 문제를 해결할 수 있을 것이라 기대한다.
"""

def update_one(var, q_idx, value):
    arr = np.array(var.value, copy=True)
    if arr.shape == ():
        var.value = value
    else:
        arr[q_idx] = value
        var.value = arr


def get_one(var, q_idx):
    arr = np.array(var.value, copy=False)
    if arr.shape == ():
        return arr.item()
    return arr[q_idx]


def optimize_ro_bayesian(
    q_idx,
    qubits,
    calibration_set,
    backend,
    ro_freq,
    ro_amp,
    ro_dur,
    acq_delay,
    acq_dur,
    x90_freq,
    x90_amp,
    pump_ch,
    pump_freq,
    pump_power,
    C_readout_dur0=0.5,
    C_acq_delay0=0.2,
    C_acq_dur0=0.5,
    MIN_DURATION_NS=200,
    MAX_DURATION_NS=10000,
    MAX_FILTER_SAMPLES=32768,
    DIGITIZER_SAMPLE_RATE_HZ=4.8e9,
    SAFETY_MARGIN=0.95,
    READOUT_DELAY_FIXED_S=0.0,
    AMP_MAX=1.0,
    N_CYCLES=30,
    N_CALLS_CYCLE=30,
    N_SHOTS=5000,

    # 개선 옵션
    N_SOBOL_INIT=128,
    N_SHOTS_PRESCAN=1000,
    N_SHOTS_BO=None,
    FAIL_LOSS=1.0,
    N_RESTARTS_OPTIMIZER=10,

    MIN_IMPROVEMENT=0.002,
    PATIENCE=2,
    BOUNDARY_EPS=0.10,
    USE_ROBUST_CENTER=True,

    freq_span=10e6,
    spec_n_steps=101,
    spec_n_shots=1000,
    metric="ks",
    save_dir="cal_plots",
    random_state=42,
    plot=True,
    live_cycle_output=True,
):
    warnings.filterwarnings("ignore", category=UserWarning)

    MAX_FILTER_TIME_S = SAFETY_MARGIN * MAX_FILTER_SAMPLES / DIGITIZER_SAMPLE_RATE_HZ
    MAX_TOTAL_TIME_S = MAX_FILTER_TIME_S

    if N_SHOTS_BO is None:
        N_SHOTS_BO = N_SHOTS
    

    HW_SOFT_FAIL_PATTERNS = (
        "Filter is too long",
        "shortest playable waveform",
    )

    progress_output = {"handle": None, "lines": []}

    def _render_progress():
        if not live_cycle_output or display is None or HTML is None:
            return
        if progress_output["handle"] is None:
            progress_output["handle"] = display(HTML("<pre></pre>"), display_id=True)
        text = "\n".join(progress_output["lines"])
        progress_output["handle"].update(HTML("<pre>" + html.escape(text) + "</pre>"))

    def _progress(line=""):
        if not live_cycle_output or display is None or HTML is None:
            print(line)
            return
        progress_output["lines"].append(line)
        _render_progress()

    def _progress_many(lines):
        if not live_cycle_output or display is None or HTML is None:
            for line in lines:
                print(line)
            return
        progress_output["lines"].extend(lines)
        _render_progress()

    def _progress_reset():
        if not live_cycle_output or display is None or HTML is None:
            return
        progress_output["lines"] = []
        _render_progress()

    def _progress_clear():
        if not live_cycle_output or display is None or HTML is None:
            return
        if progress_output["handle"] is not None:
            progress_output["handle"].update(HTML(""))
        progress_output["handle"] = None
        progress_output["lines"] = []

    def measure_kappa(target_qubit, freq_center, freq_span,
                      n_steps=101, n_shots=1000, use_pump=True, plot=True):
        freq_scan_values = np.linspace(freq_center - freq_span/2,
                                       freq_center + freq_span/2, n_steps)
        res_spec = ResonatorSpectroscopy(
            backend=backend, calibration_set=calibration_set,
            qubits=target_qubit, operation="measurement",
        )
        res_spec.configure_repetitions(
            frequencies=freq_scan_values, n_shots=n_shots,
            frequency_name="ro_freq",
        )
        if use_pump:
            synth = pump_init()
            pump_set(synth, pump_ch, pump_freq, pump_power)
            pump_on(synth, pump_ch)
        try:
            res_spec.execute()
        finally:
            if use_pump:
                pump_off(synth, pump_ch)
        if plot:
            res_spec.plot_iq()
        iq_amp = np.abs(res_spec.get_iq_array(avg=True)).squeeze()
        fit = fit_lorentzian(freq_scan_values, iq_amp, plot=plot)
        return {
            "f0": float(fit["f0"]),
            "kappa": float(fit["kappa"]),
            "Q": float(fit["Q"]),
            "fit": fit,
        }

    def estimate_initial_values(kappa_Hz, f0_Hz,
                                c_dur=C_readout_dur0,
                                c_acq_delay0=C_acq_delay0,
                                c_acq_dur0=C_acq_dur0):
        tau = 1.0 / kappa_Hz
        readout_dur0 = c_dur * tau
        acq_delay0 = c_acq_delay0 * tau
        acq_dur0 = c_acq_dur0 * tau

        lo = MIN_DURATION_NS * 1e-9
        hi = min(MAX_DURATION_NS * 1e-9, MAX_FILTER_TIME_S)
        readout_dur0 = float(np.clip(readout_dur0, lo, hi))
        acq_dur0 = float(np.clip(acq_dur0, lo, hi))
        acq_delay0 = float(max(acq_delay0, 0.0))

        return {
            "readout_frequency_Hz": float(f0_Hz),
            "readout_duration_s": float(readout_dur0),
            "acquisition_delay_s": float(acq_delay0),
            "acquisition_duration_s": float(acq_dur0),
            "readout_delay_s": float(READOUT_DELAY_FIXED_S),
            "kappa_Hz": float(kappa_Hz),
            "tau_s": float(tau),
        }

    def apply_initial_values(initial):
        update_one(ro_freq, q_idx, initial["readout_frequency_Hz"])
        update_one(ro_dur, q_idx, initial["readout_duration_s"])
        update_one(acq_delay, q_idx, initial["acquisition_delay_s"])
        update_one(acq_dur, q_idx, initial["acquisition_duration_s"])

    def print_initial_values(initial):
        qf = get_one(x90_freq, q_idx)
        qa = get_one(x90_amp, q_idx)
        _progress_many([
            "",
            "=" * 60,
            "Initial values estimated from kappa",
            "=" * 60,
            f"  qubit index       : {q_idx}",
            f"  kappa             : {initial['kappa_Hz']/1e6:.4f} MHz",
            f"  tau = 1/kappa     : {initial['tau_s']*1e9:.1f} ns",
            f"  readout frequency : {initial['readout_frequency_Hz']/1e9:.6f} GHz",
            f"  readout duration  : {initial['readout_duration_s']*1e9:.1f} ns",
            f"  acquisition delay : {initial['acquisition_delay_s']*1e9:.1f} ns",
            f"  acquisition dur.  : {initial['acquisition_duration_s']*1e9:.1f} ns",
            f"  HW filter cap     : {MAX_FILTER_TIME_S*1e9:.1f} ns",
            f"  qubit frequency   : {qf/1e9:.6f} GHz  (trusted, unchanged)",
            f"  qubit amplitude   : {qa:.6g} (arb)  (trusted, unchanged)",
        ])

    def _measure_one_point(n_shots):
        try:
            iq_experiment = IQDistributionExperiment(
                backend, calibration_set=calibration_set, qubits=qubits[q_idx]
            )
            iq_experiment.configure_repetitions(n_shots=n_shots)
            iq_experiment.backend._keep_progress_bar = False
            iq_experiment.execute()

            iq_array = iq_experiment.get_iq_array().to_numpy()
            d0 = np.vstack([np.real(iq_array[0, :, 0]), np.imag(iq_array[0, :, 0])])
            d1 = np.vstack([np.real(iq_array[0, :, 1]), np.imag(iq_array[0, :, 1])])
            X = np.hstack([d0, d1]).transpose()

            z, _, _, _ = pca_project_to_1d(X)
            gmm, labels, _ = fit_gmm_1d(z, n_components=2)
            L0 = labels[: int(len(z)/2)]
            L1 = labels[int(len(z)/2):]
            p00 = max(np.sum(L0)/len(L0), 1 - np.sum(L0)/len(L0))
            p11 = max(np.sum(L1)/len(L1), 1 - np.sum(L1)/len(L1))
            fid_gmm = (p00 + p11) / 2

            fit_result = iq_experiment.fit()
            fid_ks = list(fit_result.values())[0]["fidelity"]

            m = gmm.means_.ravel()
            ss = np.sqrt([c[0, 0] for c in gmm.covariances_])
            snr = float(np.abs((m[0] - m[1]) / np.mean(ss)))

            IQ_avg = iq_experiment.get_iq(avg=True).to_numpy()[0]
            blob = float(np.abs(IQ_avg[0] - IQ_avg[1]))

            return {
                "fid_ks": float(fid_ks),
                "fid_gmm": float(fid_gmm),
                "snr": snr,
                "blob": blob,
            }
        except RuntimeError as e:
            msg = str(e)
            if any(p in msg for p in HW_SOFT_FAIL_PATTERNS):
                _progress(
                    f"    [skip] HW limit — recording NaN  "
                    f"({msg.splitlines()[0][:80]}...)"
                )
                return {"fid_ks": np.nan, "fid_gmm": np.nan,
                        "snr": np.nan, "blob": np.nan}
            raise

    spec = measure_kappa(
        target_qubit=qubits[q_idx],
        freq_center=get_one(ro_freq, q_idx),
        freq_span=freq_span,
        n_steps=spec_n_steps,
        n_shots=spec_n_shots,
        plot=plot,
    )
    initial = estimate_initial_values(spec["kappa"], spec["f0"])
    apply_initial_values(initial)
    print_initial_values(initial)

    tau = initial["tau_s"]
    f0 = initial["readout_frequency_Hz"]

    PARAM_NAMES = ["freq", "amp", "dur", "acq_delay0", "acq_dur0"]

    ABS_BOUNDS = {
        "freq": (f0 - 5/tau, f0 + 5/tau),
        "amp": (1e-4, AMP_MAX),
        "dur": (max(200e-9, 0.5*tau), MAX_FILTER_TIME_S),
        "acq_delay0": (0.0, 5*tau),
        "acq_dur0": (max(200e-9, 0.5*tau), MAX_FILTER_TIME_S),
    }
    IS_LOG = {"freq": False, "amp": True, "dur": False,
              "acq_delay0": False, "acq_dur0": False}
    """
    ranges = {
        "freq": [f0 - 2/tau, f0 + 2/tau],
        "amp": [1e-3, 0.5],
        "dur": [max(200e-9, 1*tau), min(MAX_FILTER_TIME_S, 8*tau)],
        "acq_delay0": [0.0, 5*tau],
        "acq_dur0": [max(200e-9, 1*tau), min(MAX_FILTER_TIME_S, 8*tau)],
    }
    """
    ranges = {
        "freq": [f0 - 3/tau, f0 + 3/tau],
        "amp": [1e-3, 0.7],
        "dur": [max(200e-9, 0.7*tau), min(MAX_FILTER_TIME_S, 10*tau)],
        "acq_delay0": [0.0, 5*tau],
        "acq_dur0": [max(200e-9, 0.7*tau), min(MAX_FILTER_TIME_S, 10*tau)],
    }

    def clip_range_to_abs(name, lo, hi):
        abs_lo, abs_hi = ABS_BOUNDS[name]
        lo = max(lo, abs_lo)
        hi = min(hi, abs_hi)
        if hi <= lo:
            hi = min(abs_hi, lo + (abs_hi - abs_lo) * 0.05 + 1e-12)
            lo = max(abs_lo, hi - (abs_hi - abs_lo) * 0.05 - 1e-12)
        return [lo, hi]

    def adjust_range(name, lo, hi, best):
        w = hi - lo
        inner_lo = lo + 0.25 * w
        inner_hi = hi - 0.25 * w

        if inner_lo <= best <= inner_hi:
            new_w = w / 2
            new_lo = best - new_w / 2
            new_hi = best + new_w / 2
            action = "zoom-in"
        else:
            new_lo = best - w / 2
            new_hi = best + w / 2
            action = "shift"

        new_lo, new_hi = clip_range_to_abs(name, new_lo, new_hi)

        if action == "shift":
            moved = abs(new_lo - lo) + abs(new_hi - hi)
            if moved < 1e-12 * max(1.0, abs(lo) + abs(hi)):
                action = "zoom-in"

        return new_lo, new_hi, action

    def build_search_space(ranges):
        space = []
        for nm in PARAM_NAMES:
            lo, hi = ranges[nm]
            if IS_LOG[nm]:
                lo = max(lo, 1e-12)
                space.append(Real(lo, hi, prior="log-uniform", name=nm))
            else:
                space.append(Real(lo, hi, name=nm))
        return space

    def get_lcb_kappa(cycle):
        if cycle <= 1:
            return 3.0
        elif cycle <= 3:
            return 2.0
        elif cycle <= 6:
            return 1.5
        else:
            return 1.0
    
    def update_global_best_from_log(call_log, prefer_high_shot=True): #log기반으로 global best update
        good = [
            c for c in call_log
            if not c.get("failed", False)
            and np.isfinite(c.get("fidelity", np.nan))
        ]

        if not good:
            return None, -np.inf

        # pre-scan은 low-shot이므로, 본 BO 측정이 있으면 cycle >= 1만 final best 후보로 사용
        if prefer_high_shot:
            high_shot_good = [c for c in good if c.get("cycle", 0) >= 1]
            pool = high_shot_good if high_shot_good else good
        else:
            pool = good

        best = max(pool, key=lambda c: c["fidelity"])
        return list(map(float, best["params"])), float(best["fidelity"])
    
    def get_robust_cycle_center(call_log, cycle_idx, top_k=5): #top k구현
        good = [
            c for c in call_log
            if c.get("cycle") == cycle_idx
            and not c.get("failed", False)
            and np.isfinite(c.get("fidelity", np.nan))
        ]

        if not good:
            return None

        good = sorted(good, key=lambda c: c["fidelity"], reverse=True)
        top = good[:min(top_k, len(good))]

        fids = np.array([c["fidelity"] for c in top])
        weights = fids - np.min(fids) + 1e-6
        weights = weights / np.sum(weights)

        params = np.array([c["params"] for c in top])
        center = np.sum(params * weights[:, None], axis=0)

        return center.tolist()
    
    MIN_WIDTH = {   #종료조건 추가, 이 이상 range가 작아져가 수렴판정
        "freq": 0.2 / tau,
        "amp": 0.005,
        "dur": 50e-9,
        "acq_delay0": 20e-9,
        "acq_dur0": 50e-9,
    }


    def range_width_small_enough(ranges):
        for nm in PARAM_NAMES:
            lo, hi = ranges[nm]
            if (hi - lo) > MIN_WIDTH[nm]:
                return False
        return True
    
    def any_center_near_boundary(center, ranges, eps=BOUNDARY_EPS): #10%내로 boundary랑 가까이 있는지 확인
        for i, nm in enumerate(PARAM_NAMES):
            lo, hi = ranges[nm]
            w = hi - lo
            x = center[i]

            if x <= lo + eps * w or x >= hi - eps * w:
                return True

        return False
    

    def expand_ranges_around_best(ranges, best_params, factor=1.5):
        new_ranges = {}

        for i, nm in enumerate(PARAM_NAMES):
            lo, hi = ranges[nm]
            center = best_params[i]
            old_w = hi - lo
            new_w = old_w * factor

            new_lo = center - new_w / 2
            new_hi = center + new_w / 2

            new_ranges[nm] = clip_range_to_abs(nm, new_lo, new_hi)

        return new_ranges
    
    def expand_ranges_around_best(ranges, best_params, factor=1.5):
        new_ranges = {}

        for i, nm in enumerate(PARAM_NAMES):
            lo, hi = ranges[nm]
            center = best_params[i]
            old_w = hi - lo
            new_w = old_w * factor

            new_lo = center - new_w / 2
            new_hi = center + new_w / 2

            new_ranges[nm] = clip_range_to_abs(nm, new_lo, new_hi)

        return new_ranges





    all_call_log = []

    def make_objective(cycle_idx, n_calls_cycle, n_shots):
        local = {"count": 0}

        def _log_failed(params, reason, fidelity_value=np.nan):
            all_call_log.append({
                "cycle": cycle_idx,
                "params": list(map(float, params)),
                "fidelity": float(fidelity_value) if np.isfinite(fidelity_value) else float("nan"),
                "fid_ks": np.nan,
                "fid_gmm": np.nan,
                "snr": np.nan,
                "blob": np.nan,
                "failed": True,
                "reason": reason,
                "n_shots": int(n_shots),
            })

        def objective(params):
            freq, amp, dur, acq_delay0, acq_dur0 = params

            # --- Hard 제약 체크 ---
            if amp > AMP_MAX:
                _log_failed(params, "amp_over_max")
                return FAIL_LOSS

            if dur > MAX_FILTER_TIME_S or acq_dur0 > MAX_FILTER_TIME_S:
                _log_failed(params, "filter_time_over_max")
                return FAIL_LOSS

            if acq_delay0 + acq_dur0 > MAX_TOTAL_TIME_S:
                _progress(
                    f"  [skip] total time {(acq_delay0+acq_dur0)*1e9:.0f}ns "
                    f"> {MAX_TOTAL_TIME_S*1e9:.0f}ns"
                )
                _log_failed(params, "total_time_over_max")
                return FAIL_LOSS

            # --- 파라미터 세팅 ---
            update_one(ro_freq, q_idx, freq)
            update_one(ro_amp, q_idx, amp)
            update_one(ro_dur, q_idx, dur)
            update_one(acq_delay, q_idx, acq_delay0)
            update_one(acq_dur, q_idx, acq_dur0)

            # --- 측정 ---
            try:
                m = _measure_one_point(n_shots=n_shots)

                if metric == "ks":
                    fidelity = m["fid_ks"]
                elif metric == "gmm":
                    fidelity = m["fid_gmm"]
                else:
                    raise ValueError("metric must be 'ks' or 'gmm'")

            except Exception as e:
                _progress(f"  FAILED: {type(e).__name__}: {repr(e)}")
                _log_failed(params, f"exception:{type(e).__name__}", fidelity_value=0.0)
                return FAIL_LOSS

            # --- NaN 가드 ---
            if not np.isfinite(fidelity):
                _progress("  NaN fidelity (HW limit) — penalizing")
                _log_failed(params, "nan_fidelity")
                return FAIL_LOSS

            local["count"] += 1

            all_call_log.append({
                "cycle": cycle_idx,
                "params": list(map(float, params)),
                "fidelity": float(fidelity),
                "fid_ks": float(m["fid_ks"]),
                "fid_gmm": float(m["fid_gmm"]),
                "snr": float(m["snr"]),
                "blob": float(m["blob"]),
                "failed": False,
                "n_shots": int(n_shots),
            })

            line = (
                f"  [C{cycle_idx} {local['count']:2d}/{n_calls_cycle}] "
                f"shots={n_shots} "
                f"freq={freq/1e9:.5f}GHz amp={amp:.4f} "
                f"dur={dur*1e9:.0f}ns "
                f"acq_del={acq_delay0*1e9:.0f}ns "
                f"acq_dur0={acq_dur0*1e9:.0f}ns "
                f"→ fid={fidelity:.4f}"
            )
            _progress(line)

            return -fidelity

        return objective

    _progress_many([
        "",
        "=" * 60,
        f"Multi-cycle Bayesian optimization for q_idx={q_idx} "
        f"(max {N_CYCLES} cycles × {N_CALLS_CYCLE} calls)",
        "=" * 60,
    ])

    best_fidelity_global = -np.inf
    best_params_global = None
    warm_x, warm_y = [], []
    stop_reason = f"reached max cycles ({N_CYCLES})"
    no_improve_count = 0

    synth = pump_init()
    pump_set(synth, pump_ch, pump_freq, pump_power)
    pump_on(synth, pump_ch)
        
    try:
        if N_SOBOL_INIT > 0:
            _progress_many([
                "",
                "=" * 60,
                f"Sobol pre-scan: {N_SOBOL_INIT} points × {N_SHOTS_PRESCAN} shots",
                "=" * 60,
            ])

            space = build_search_space(ranges)

            prescan_objective = make_objective(
                cycle_idx=0,
                n_calls_cycle=N_SOBOL_INIT,
                n_shots=N_SHOTS_PRESCAN,
            )

            sampler = Sobol()
            sobol_points = sampler.generate(
                dimensions=space,
                n_samples=N_SOBOL_INIT,
                random_state=random_state,
            )

            for p in sobol_points:
                prescan_objective(p)

            warm_x = [c["params"] for c in all_call_log if not c["failed"]]
            warm_y = [-c["fidelity"] for c in all_call_log if not c["failed"]]

            best_params_global, best_fidelity_global = update_global_best_from_log(
                all_call_log,
                prefer_high_shot=False,
            )

            if best_params_global is not None:
                _progress(
                    f"\n[Pre-scan] best fid = {best_fidelity_global:.4f} "
                    f"from {len(warm_x)} valid points"
                )
            else:
                _progress("\n[Pre-scan] no valid point found")

        for cycle in range(1, N_CYCLES + 1):
            if cycle > 1:
                _progress_reset()

            cycle_lines = [
                "",
                "=" * 60,
                f"CYCLE {cycle}",
                "=" * 60,
            ]
            for nm in PARAM_NAMES:
                cycle_lines.append(f"  range {nm:10s}: [{ranges[nm][0]:.6g}, {ranges[nm][1]:.6g}]")
            _progress_many(cycle_lines)

            space = build_search_space(ranges)

            x0c, y0c = [], []
            for x, y in zip(warm_x, warm_y):
                if all(space[i].low <= x[i] <= space[i].high for i in range(len(space))):
                    x0c.append(x)
                    y0c.append(y)

            if best_params_global is not None:
                if all(space[i].low <= best_params_global[i] <= space[i].high
                    for i in range(len(space))):
                    x0c.append(best_params_global)
                    y0c.append(-best_fidelity_global)

            objective = make_objective(
                cycle_idx=cycle,
                n_calls_cycle=N_CALLS_CYCLE,
                n_shots=N_SHOTS_BO,
            )

            kappa = get_lcb_kappa(cycle)

            if x0c:
                result = gp_minimize(
                    objective,
                    space,
                    n_calls=N_CALLS_CYCLE,
                    x0=x0c,
                    y0=y0c,
                    n_initial_points=0,
                    acq_func="LCB",
                    kappa=kappa,
                    acq_optimizer="lbfgs",
                    n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
                    noise="gaussian",
                    random_state=random_state + cycle,
                )
            else:
                result = gp_minimize(
                    objective,
                    space,
                    n_calls=N_CALLS_CYCLE,
                    n_initial_points=max(10, N_CALLS_CYCLE // 3),
                    initial_point_generator="sobol",
                    acq_func="LCB",
                    kappa=kappa,
                    acq_optimizer="lbfgs",
                    n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
                    noise="gaussian",
                    random_state=random_state + cycle,
                )

            cyc_best_fid = -result.fun
            cyc_best = result.x

            prev_best_fid = best_fidelity_global

            best_params_global, best_fidelity_global = update_global_best_from_log(
                all_call_log,
                prefer_high_shot=True,
            )

            improvement = best_fidelity_global - prev_best_fid

            _progress(f"")
            _progress(f"  [Cycle {cycle}] best fid = {cyc_best_fid:.4f} "
                      f"(global best = {best_fidelity_global:.4f}, "
                      f"Δ = {improvement:+.4f})")

            warm_x = [c["params"] for c in all_call_log if not c["failed"]]
            warm_y = [-c["fidelity"] for c in all_call_log if not c["failed"]]

            _progress("")
            _progress(f"  --- 범위 조정 (cycle {cycle} → {cycle+1}) ---")
            
            if USE_ROBUST_CENTER:
                robust_center = get_robust_cycle_center(
                    all_call_log,
                    cycle_idx=cycle,
                    top_k=5,
                )
            else:
                robust_center = None

            if robust_center is not None:
                range_center = robust_center
            else:
                range_center = cyc_best

            bf = dict(zip(PARAM_NAMES, range_center))

            actions = {}
            for nm in PARAM_NAMES:
                lo, hi = ranges[nm]
                new_lo, new_hi, action = adjust_range(nm, lo, hi, bf[nm])
                ranges[nm] = [new_lo, new_hi]
                actions[nm] = action
                _progress(f"    {nm:10s} center={bf[nm]:.6g}  {action:8s}"
                          f"→ [{new_lo:.6g}, {new_hi:.6g}]")
                
            
            if improvement < MIN_IMPROVEMENT:
                no_improve_count += 1
            else:
                no_improve_count = 0
                
            

            shifted = [nm for nm, a in actions.items() if a == "shift"]
            all_zoom = len(shifted) == 0

            small_range = range_width_small_enough(ranges)
            near_boundary = any_center_near_boundary(range_center, ranges)

            if (
                all_zoom
                and small_range
                and not near_boundary
                and no_improve_count >= PATIENCE
            ):
                stop_reason = (
                    f"converged: all zoom-in, small range, "
                    f"no improvement for {PATIENCE} cycles, "
                    f"center not near boundary"
                )
                _progress("")
                _progress(f"  {stop_reason}")
                break

            if no_improve_count >= PATIENCE and (near_boundary or not small_range):
                _progress("")
                _progress("  개선 정체지만 range가 넓거나 center가 경계 근처 → range 재확장")
                ranges = expand_ranges_around_best(
                    ranges,
                    best_params_global,
                    factor=1.5,
                )
                no_improve_count = 0
                continue

            if shifted:
                _progress("")
                _progress(f"  아직 평행이동 중인 파라미터 {len(shifted)}개: {shifted} "
                        f"→ 다음 사이클 계속")
            else:
                _progress("")
                _progress("  모든 축 zoom-in이지만 종료 조건 미충족 → 다음 사이클 계속")
        else:
            _progress("")
            _progress(f"  최대 사이클({N_CYCLES}) 도달 → 종료")

    finally:
        pump_off(synth, pump_ch)

    _progress_clear()

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"  qubit index       : {q_idx}")
    print(f"  종료 사유          : {stop_reason}")

    if best_params_global is None:
        raise RuntimeError("No valid optimization point was measured.")

    bp = best_params_global
    print(f"  Best fidelity     : {best_fidelity_global:.4f}")
    print(f"  readout freq      : {bp[0]/1e9:.6f} GHz")
    print(f"  readout amplitude : {bp[1]:.5f}")
    print(f"  readout duration  : {bp[2]*1e9:.1f} ns")
    print(f"  acquisition delay : {bp[3]*1e9:.1f} ns")
    print(f"  acquisition dur.  : {bp[4]*1e9:.1f} ns")
    print(f"  (acq_delay0+acq_dur0 = {(bp[3]+bp[4])*1e9:.1f} ns / "
          f"limit {MAX_TOTAL_TIME_S*1e9:.0f} ns)")

    update_one(ro_freq, q_idx, bp[0])
    update_one(ro_amp, q_idx, bp[1])
    update_one(ro_dur, q_idx, bp[2])
    update_one(acq_delay, q_idx, bp[3])
    update_one(acq_dur, q_idx, bp[4])

    os.makedirs(save_dir, exist_ok=True)
    save_name = f"bayesian_multicycle_results_q{q_idx}.json"
    save_path = os.path.join(save_dir, save_name)

    with open(save_path, "w") as fp:
        json.dump({
            "q_idx": q_idx,
            "metric": metric,
            "kappa_Hz": spec["kappa"],
            "f0_Hz": spec["f0"],
            "best_params": list(map(float, bp)),
            "best_fidelity": float(best_fidelity_global),
            "stop_reason": stop_reason,
            "n_cycles_run": max([c["cycle"] for c in all_call_log], default=0),
            "N_SOBOL_INIT": N_SOBOL_INIT,
            "N_SHOTS_PRESCAN": N_SHOTS_PRESCAN,
            "N_SHOTS_BO": N_SHOTS_BO,
            "acq_func": "LCB",
            "N_RESTARTS_OPTIMIZER": N_RESTARTS_OPTIMIZER,
            "call_log": all_call_log,
        }, fp, indent=2)

    print(f"\nResults saved to {save_path}")

    good = [c for c in all_call_log if not c["failed"]]
    fids = [c["fidelity"] for c in good]
    cycles_of = [c["cycle"] for c in good]
    best_so_far = np.maximum.accumulate(fids) if fids else []

    if plot:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(range(1, len(fids)+1), fids, "o-", alpha=0.4, label="each measurement")
        ax.plot(range(1, len(fids)+1), best_so_far, "s-", label="best so far")
        if cycles_of:
            for c in range(1, max(cycles_of) + 1):
                if c in cycles_of:
                    first = next(i for i, cc in enumerate(cycles_of) if cc == c)
                    ax.axvline(first + 0.5, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("measurement # (all cycles)")
        ax.set_ylabel("fidelity")
        ax.set_title(f"Multi-cycle Bayesian optimization q{q_idx}")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"bayesian_multicycle_q{q_idx}.png")
        plt.savefig(plot_path, dpi=120)
        plt.show()
    else:
        plot_path = None

    return {
        "q_idx": q_idx,
        "metric": metric,
        "spec": spec,
        "initial": initial,
        "best_params": bp,
        "best_fidelity": best_fidelity_global,
        "stop_reason": stop_reason,
        "call_log": all_call_log,
        "ranges_final": ranges,
        "save_path": save_path,
        "plot_path": plot_path,
    }
