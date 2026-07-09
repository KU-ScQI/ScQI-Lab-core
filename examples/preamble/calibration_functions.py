import numpy as np
import time
import warnings
from IPython.display import clear_output
from Preamble.fitter import *
from Preamble.pump_utils import *
import matplotlib.pyplot as plt
import keysight.qcs as qcs
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from preamble.mapper_utils import *
from preamble.calibration_utils import *
from preamble.pump_utils import *
from preamble.fitter import *
from preamble.functions import *
from preamble.readout_utils import *
from preamble.pca_gmm_utils import *

def to_numpy(value):
    try:
        value = value.to_numpy()
    except Exception:
        pass
    return np.asarray(value)

def to_scalar(value, index=0):
    arr = to_numpy(value).squeeze()
    if arr.shape == ():
        return float(arr)
    return float(arr[index])

def validate_unit_amplitude(value, name="amplitude"):
    value = float(value)
    if not (-1.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between -1 and 1. got {value}")
    return value

def execute_program(
    program,
    backend,
    pump=None,
    pump_ch=1,
    pump_freq=None,
    pump_power=None,
):
    if pump is not None:
        pump.connect()
        if pump_freq is not None or pump_power is not None:
            pump.set_on(ch=pump_ch, freq=pump_freq, power=pump_power)
        else:
            pump.on(ch=pump_ch)

    try:
        return qcs.Executor(backend).execute(program)
    finally:
        if pump is not None:
            pump.off(ch=pump_ch).close_all()


def get_iq_signal(iq_array, iq_mode):
    iq = to_numpy(iq_array).squeeze()
    iq_mode = iq_mode.lower()

    if iq_mode == "phase":
        return np.unwrap(np.angle(iq)).squeeze()
    if iq_mode == "i":
        return np.real(iq).squeeze()
    if iq_mode == "q":
        return np.imag(iq).squeeze()
    if iq_mode == "abs":
        return np.abs(iq).squeeze()
    if iq_mode == "pca":
        return project_iq(iq).squeeze()

    raise ValueError("iq_mode must be 'phase', 'i', 'q', 'abs', or 'pca'")


def measure_ro_freqs(
    f_centers,
    backend,
    calibration_set,
    qubits,
    pump_ch,
    pump_freq,
    pump_power,
    f_range,
    n_steps,
    n_shots,
    q_indices=None,
    clear=False,
    iq_mode="abs",
    pump=None,  # pump는 밖에서 입력받음
):
    ro_amp = calibration_set.variables.ro_amp
    ro_freq = calibration_set.variables.ro_freq

    f_centers = np.asarray(f_centers, dtype=float)

    if f_centers.shape == ():
        f_centers = np.array([float(f_centers)], dtype=float)

    warnings.filterwarnings("ignore", message=r".*was rounded.*")

    if q_indices is None:
        q_indices = range(len(qubits.labels))
    q_indices = list(q_indices)

    if len(f_centers) == 1 and len(q_indices) > 1:
        f_centers = np.repeat(f_centers[0], len(q_indices))

    if len(f_centers) != len(q_indices):
        raise ValueError(
            f"len(f_centers) must be 1 or len(q_indices). "
            f"got len(f_centers)={len(f_centers)}, len(q_indices)={len(q_indices)}"
        )

    ro_freqs = np.zeros(len(q_indices), dtype=float)
    program_compileds = []

    pump_is_on = False

    try:
        if pump is not None:
            pump.connect()
            pump.set_on(
                ch=pump_ch,
                freq=pump_freq,
                power=pump_power,
            )
            pump_is_on = True

        for i, q_idx in enumerate(q_indices):
            f_center = float(f_centers[i])

            ro_amp_values = np.asarray(ro_amp.value, dtype=float)

            if ro_amp_values.shape == ():
                current_ro_amp = float(ro_amp_values)
            else:
                current_ro_amp = float(ro_amp_values[q_idx])

            print(f"Q{q_idx} ro_freq progress: {i + 1}/{len(q_indices)}")
            print(f"Q{q_idx} current ro_amp        : {current_ro_amp:.6g}")
            print(f"Q{q_idx} ro_freq sweep center  : {f_center / GHz:.9f} GHz")
            print(f"Q{q_idx} ro_freq bandwidth     : {f_range / MHz:.3f} MHz")

            freq_scan_values = np.linspace(
                f_center - f_range / 2,
                f_center + f_range / 2,
                n_steps,
            )

            target_qubit = qubits[q_idx]

            ro_freq_list = qcs.Array(
                f"ro_freq_list_q{q_idx}",
                value=freq_scan_values,
                dtype=float,
                unit="Hz",
            )

            program = qcs.Program()

            program.add_measurement(
                target_qubit,
                new_layer=True,
            )

            program = program.sweep(
                ro_freq_list,
                ro_freq[q_idx],
            )

            linker_pass = qcs.LinkerPass(
                *calibration_set.linkers.values()
            )

            program_compiled = linker_pass.apply(program)
            program_compiled.n_shots(n_shots)

            program_compiled = qcs.Executor(backend).execute(program_compiled)

            iq_signal = get_iq_signal(
                program_compiled.get_iq_array(avg=True),
                iq_mode,
            )

            iq_signal = np.asarray(iq_signal).squeeze().reshape(-1)

            fit_result = fit_lorentzian(
                freq_scan_values,
                iq_signal,
                plot=False,
            )

            ro_freqs[i] = float(fit_result["f0"])
            program_compileds.append(program_compiled)

            print(f"Q{q_idx} fitted ro_freq : {ro_freqs[i] / GHz:.9f} GHz")

            if clear:
                plt.close("all")
                clear_output(wait=False)

    finally:
        if pump is not None and pump_is_on:
            pump.off(pump_ch)

    return ro_freqs, program_compileds


def measure_ro_amp_sweep(
    f_centers,
    backend,
    calibration_set,
    qubits,
    pump_ch,
    pump_freq,
    pump_power,
    f_range,
    n_freq_steps,
    n_shots,
    start_dbm=-60,
    end_dbm=-40,
    n_amp_steps=7,
    q_indices=None,
    clear=False,
    iq_mode="abs",
    pump=None,    #pump는 밖에서 입력받음
    restore_ro_amp=True,
):
    """
    Readout amplitude punchout sweep.

    각 ro_amp마다 ro_freq sweep을 수행해서
    frequency x amplitude response matrix를 만든다.
    """

    ro_amp = calibration_set.variables.ro_amp
    ro_freq = calibration_set.variables.ro_freq

    f_centers = np.asarray(f_centers, dtype=float)

    if f_centers.shape == ():
        f_centers = np.array([float(f_centers)], dtype=float)

    warnings.filterwarnings("ignore", message=r".*was rounded.*")

    if q_indices is None:
        q_indices = range(len(qubits.labels))
    q_indices = list(q_indices)

    if len(f_centers) == 1 and len(q_indices) > 1:
        f_centers = np.repeat(f_centers[0], len(q_indices))

    if len(f_centers) != len(q_indices):
        raise ValueError(
            f"len(f_centers) must be 1 or len(q_indices). "
            f"got len(f_centers)={len(f_centers)}, len(q_indices)={len(q_indices)}"
        )

    dbm_list = np.linspace(start_dbm, end_dbm, n_amp_steps)
    amp_values = 10 ** (dbm_list / 20)

    if np.any(np.abs(amp_values) > 1.0):
        raise ValueError(
            "Generated ro_amp contains value outside [-1, 1]. "
            f"amp range = [{amp_values.min()}, {amp_values.max()}]"
        )

    original_ro_amp_values = np.asarray(ro_amp.value, dtype=float).copy()

    def set_ro_amp(q_idx, amp):
        amp = float(amp)

        if not (-1.0 <= amp <= 1.0):
            raise ValueError(f"ro_amp must be between -1 and 1. got {amp}")

        ro_amp_values = np.asarray(ro_amp.value, dtype=float).copy()

        if ro_amp_values.shape == ():
            ro_amp.value = amp
        else:
            ro_amp_values[q_idx] = amp
            ro_amp.value = ro_amp_values

    freq_axes = []
    signal_matrices = []
    program_compileds_all = []

    pump_is_on = False

    try:
        if pump is not None:
            pump.connect()
            pump.set_on(
                ch=pump_ch,
                freq=pump_freq,
                power=pump_power,
            )
            pump_is_on = True

        for i, q_idx in enumerate(q_indices):
            f_center = float(f_centers[i])

            print(f"Q{q_idx} ro_amp punchout progress: {i + 1}/{len(q_indices)}")
            print(f"Q{q_idx} ro_amp sweep center    : {f_center / GHz:.9f} GHz")
            print(f"Q{q_idx} ro_amp sweep bandwidth : {f_range / MHz:.3f} MHz")
            print(f"Q{q_idx} ro_amp dBm range       : {start_dbm:.1f} ~ {end_dbm:.1f} dBm")
            print(f"Q{q_idx} ro_amp points          : {n_amp_steps}")

            freq_scan_values = np.linspace(
                f_center - f_range / 2,
                f_center + f_range / 2,
                n_freq_steps,
            )

            target_qubit = qubits[q_idx]

            signal_matrix = np.zeros(
                (n_freq_steps, n_amp_steps),
                dtype=float,
            )


            program_compileds = []

            for amp_i, amp in enumerate(amp_values):
                dbm = float(dbm_list[amp_i])

                print(
                    f"  Q{q_idx} amp sweep "
                    f"{amp_i + 1}/{n_amp_steps} | "
                    f"{dbm:.1f} dBm | amp={amp:.6g}"
                )

                set_ro_amp(q_idx, amp)

                ro_freq_list = qcs.Array(
                    f"ro_amp_q{q_idx}_freq_list_amp{amp_i}",
                    value=freq_scan_values,
                    dtype=float,
                    unit="Hz",
                )

                program = qcs.Program()

                program.add_measurement(
                    target_qubit,
                    new_layer=True,
                )

                program = program.sweep(
                    ro_freq_list,
                    ro_freq[q_idx],
                )

                linker_pass = qcs.LinkerPass(
                    *calibration_set.linkers.values()
                )

                program_compiled = linker_pass.apply(program)
                program_compiled.n_shots(n_shots)

                program_compiled = qcs.Executor(backend).execute(program_compiled)

                iq_array = program_compiled.get_iq_array(avg=True)

                signal = get_iq_signal(
                    iq_array,
                    iq_mode,
                )

                signal = np.asarray(signal).squeeze().reshape(-1)

                if len(signal) != len(freq_scan_values):
                    raise RuntimeError(
                        f"Q{q_idx} ro_amp signal length mismatch at "
                        f"amp_idx={amp_i}: "
                        f"len(signal)={len(signal)}, "
                        f"len(freq_scan_values)={len(freq_scan_values)}"
                    )

                signal_matrix[:, amp_i] = signal

                program_compileds.append(program_compiled)


            freq_axes.append(freq_scan_values)
            signal_matrices.append(signal_matrix)
            program_compileds_all.append(program_compileds)

            if clear:
                plt.close("all")
                clear_output(wait=False)

    finally:
        if restore_ro_amp:
            ro_amp.value = original_ro_amp_values

        if pump is not None and pump_is_on:
            pump.off(pump_ch)

    return {
        "freqs": freq_axes,
        "dbm_list": dbm_list,
        "amp_values": amp_values,
        "signals": signal_matrices,
        "programs": program_compileds_all,
    }

def measure_qubit_freqs(
    f_centers,
    backend,
    calibration_set,
    qubits,
    pump_ch,
    pump_freq,
    pump_power,
    f_range,
    n_steps,
    n_shots,
    q_indices=None,
    iq_mode="abs",
    clear=False,
    pump=None,
):
    
    x90_freq = calibration_set.variables.x90_freq
    x90_amp = calibration_set.variables.x90_amp

    f_centers = np.asarray(f_centers, dtype=float)

    if f_centers.shape == ():
        f_centers = np.array([float(f_centers)], dtype=float)

    warnings.filterwarnings("ignore", message=r".*was rounded.*")

    if q_indices is None:
        q_indices = range(len(qubits.labels))
    q_indices = list(q_indices)

    if len(f_centers) == 1 and len(q_indices) > 1:
        f_centers = np.repeat(f_centers[0], len(q_indices))

    if len(f_centers) != len(q_indices):
        raise ValueError(
            f"len(f_centers) must be 1 or len(q_indices). "
            f"got len(f_centers)={len(f_centers)}, len(q_indices)={len(q_indices)}"
        )

    qubit_freqs = np.zeros(len(q_indices), dtype=float)
    program_compileds = []

    pump_is_on = False

    try:
        if pump is not None:
            pump.connect()
            pump.set_on(
                ch=pump_ch,
                freq=pump_freq,
                power=pump_power,
            )
            pump_is_on = True

        for i, q_idx in enumerate(q_indices):
            f_center = float(f_centers[i])

            x90_amp_values = np.asarray(x90_amp.value, dtype=float)

            if x90_amp_values.shape == ():
                current_x90_amp = float(x90_amp_values)
            else:
                current_x90_amp = float(x90_amp_values[q_idx])

            print(f"Q{q_idx} qubit_freq progress: {i + 1}/{len(q_indices)}")
            print(f"Q{q_idx} qubit_freq sweep center    : {f_center / GHz:.9f} GHz")
            print(f"Q{q_idx} qubit_freq sweep bandwidth : {f_range / MHz:.3f} MHz")
            print(f"Q{q_idx} current x90_amp            : {current_x90_amp:.6g}")

            freq_scan_values = np.linspace(
                f_center - f_range / 2,
                f_center + f_range / 2,
                n_steps,
            )

            target_qubit = qubits[q_idx]

            qubit_freq_list = qcs.Array(
                f"qubit_freq_list_q{q_idx}",
                value=freq_scan_values,
                dtype=float,
                unit="Hz",
            )

            program = qcs.Program()

            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
            )

            program.add_measurement(
                target_qubit,
                new_layer=True,
            )

            program = program.sweep(
                qubit_freq_list,
                x90_freq[q_idx],
            )

            linker_pass = qcs.LinkerPass(
                *calibration_set.linkers.values()
            )

            program_compiled = linker_pass.apply(program)
            program_compiled.n_shots(n_shots)

            program_compiled = qcs.Executor(backend).execute(program_compiled)

            iq_signal = get_iq_signal(
                program_compiled.get_iq_array(avg=True),
                iq_mode,
            )

            iq_signal = np.asarray(iq_signal).squeeze().reshape(-1)

            fit_result = fit_lorentzian(
                freq_scan_values,
                iq_signal,
                plot=False,
            )

            qubit_freqs[i] = float(fit_result["f0"])
            program_compileds.append(program_compiled)

            print(f"Q{q_idx} fitted qubit_freq : {qubit_freqs[i] / GHz:.9f} GHz")

            if clear:
                plt.close("all")
                clear_output(wait=False)

    finally:
        if pump is not None and pump_is_on:
            pump.off(pump_ch)

    return qubit_freqs, program_compileds



def measure_rabi_amps(
    amp_ranges,
    backend,
    calibration_set,
    qubits,
    pump_ch,
    pump_freq,
    pump_power,
    n_steps,
    n_shots,
    q_indices=None,
    iq_mode="phase",
    clear=False,
    pump=None,
):
    x90_amp = calibration_set.variables.x90_amp
    x90_freq = calibration_set.variables.x90_freq

    amp_ranges = np.asarray(amp_ranges, dtype=float)

    if amp_ranges.ndim == 1:
        amp_ranges = amp_ranges.reshape(1, -1)

    warnings.filterwarnings("ignore", message=r".*was rounded.*")

    if q_indices is None:
        q_indices = range(len(qubits.labels))
    q_indices = list(q_indices)

    if amp_ranges.shape[0] == 1 and len(q_indices) > 1:
        amp_ranges = np.repeat(amp_ranges, len(q_indices), axis=0)

    if amp_ranges.shape[0] != len(q_indices):
        raise ValueError(
            f"amp_ranges must have 1 row or len(q_indices) rows. "
            f"got amp_ranges.shape[0]={amp_ranges.shape[0]}, "
            f"len(q_indices)={len(q_indices)}"
        )

    rabi_amps = np.zeros(len(q_indices), dtype=float)
    program_compileds = []

    pump_is_on = False

    try:
        if pump is not None:
            pump.connect()
            pump.set_on(
                ch=pump_ch,
                freq=pump_freq,
                power=pump_power,
            )
            pump_is_on = True

        for i, q_idx in enumerate(q_indices):
            amp_range = amp_ranges[i]

            amp_scan_values = np.linspace(
                amp_range[0],
                amp_range[-1],
                n_steps,
            )

            if np.any(np.abs(amp_scan_values) > 1.0):
                raise ValueError(
                    f"Q{q_idx} x90_amp sweep contains value outside [-1, 1]. "
                    f"amp range = [{amp_scan_values.min()}, {amp_scan_values.max()}]"
                )

            x90_freq_values = np.asarray(x90_freq.value, dtype=float)
            x90_amp_values = np.asarray(x90_amp.value, dtype=float)

            if x90_freq_values.shape == ():
                current_x90_freq = float(x90_freq_values)
            else:
                current_x90_freq = float(x90_freq_values[q_idx])

            if x90_amp_values.shape == ():
                current_x90_amp = float(x90_amp_values)
            else:
                current_x90_amp = float(x90_amp_values[q_idx])

            print(f"Q{q_idx} rabi_amp progress       : {i + 1}/{len(q_indices)}")
            print(f"Q{q_idx} current x90_freq        : {current_x90_freq / GHz:.9f} GHz")
            print(f"Q{q_idx} current x90_amp         : {current_x90_amp:.6g}")
            print(f"Q{q_idx} x90_amp sweep range     : {amp_scan_values[0]:.6g} ~ {amp_scan_values[-1]:.6g}")
            print(f"Q{q_idx} x90_amp sweep points    : {n_steps}")

            target_qubit = qubits[q_idx]

            x90_amp_list = qcs.Array(
                f"x90_amp_list_q{q_idx}",
                value=amp_scan_values,
                dtype=float,
                unit="",
            )

            program = qcs.Program()

            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
                new_layer=True,
            )

            program.add_measurement(
                target_qubit,
                new_layer=True,
            )

            program = program.sweep(
                x90_amp_list,
                x90_amp[q_idx],
            )

            linker_pass = qcs.LinkerPass(
                *calibration_set.linkers.values()
            )

            program_compiled = linker_pass.apply(program)
            program_compiled.n_shots(n_shots)

            program_compiled = qcs.Executor(backend).execute(program_compiled)

            iq_signal = get_iq_signal(
                program_compiled.get_iq_array(avg=True),
                iq_mode,
            )

            iq_signal = np.asarray(iq_signal).squeeze().reshape(-1)

            if len(iq_signal) != len(amp_scan_values):
                raise RuntimeError(
                    f"Q{q_idx} rabi signal length mismatch: "
                    f"len(iq_signal)={len(iq_signal)}, "
                    f"len(amp_scan_values)={len(amp_scan_values)}"
                )

            _, rabi_amp = amplitude_rabi_fit(
                amp_scan_values,
                iq_signal,
            )

            rabi_amps[i] = float(rabi_amp)
            program_compileds.append(program_compiled)

            print(f"Q{q_idx} fitted x90_amp          : {rabi_amps[i]:.9g}")

            if clear:
                plt.close("all")
                clear_output(wait=False)

    finally:
        if pump is not None and pump_is_on:
            pump.off(pump_ch)

    return rabi_amps, program_compileds

def measure_anharmonicity(
    f_centers,
    backend,
    calibration_set,
    qubits,
    pump_ch,
    pump_freq,
    pump_power,
    f_range,
    n_steps,
    n_shots,
    x90_spectroscopy_amp=None,
    expected_anharmonicity=250 * MHz,
    q_indices=None,
    iq_mode="abs",
    plot=False,
    clear=False,
    pump=None,
    restore_x90_amp=True,
):
    """
    Strong-drive qubit spectroscopy 기반 anharmonicity 측정.

        anharmonicity = 2 * (ge_freq - gf_freq)
    """

    x90_freq = calibration_set.variables.x90_freq
    x90_amp = calibration_set.variables.x90_amp

    f_centers = np.asarray(f_centers, dtype=float)

    if f_centers.shape == ():
        f_centers = np.array([float(f_centers)], dtype=float)

    warnings.filterwarnings("ignore", message=r".*was rounded.*")

    if q_indices is None:
        q_indices = range(len(qubits.labels))
    q_indices = list(q_indices)

    if len(f_centers) == 1 and len(q_indices) > 1:
        f_centers = np.repeat(f_centers[0], len(q_indices))

    if len(f_centers) != len(q_indices):
        raise ValueError(
            f"len(f_centers) must be 1 or len(q_indices). "
            f"got len(f_centers)={len(f_centers)}, len(q_indices)={len(q_indices)}"
        )

    if x90_spectroscopy_amp is not None:
        x90_spectroscopy_amp = np.asarray(x90_spectroscopy_amp, dtype=float)

        if x90_spectroscopy_amp.shape == ():
            x90_spectroscopy_amp = np.array(
                [float(x90_spectroscopy_amp)],
                dtype=float,
            )

        if len(x90_spectroscopy_amp) == 1 and len(q_indices) > 1:
            x90_spectroscopy_amp = np.repeat(
                x90_spectroscopy_amp[0],
                len(q_indices),
            )

        if len(x90_spectroscopy_amp) != len(q_indices):
            raise ValueError(
                f"len(x90_spectroscopy_amp) must be 1 or len(q_indices). "
                f"got len(x90_spectroscopy_amp)={len(x90_spectroscopy_amp)}, "
                f"len(q_indices)={len(q_indices)}"
            )

    original_x90_amp_values = np.asarray(x90_amp.value, dtype=float).copy()

    def get_var_value(var, q_idx):
        values = np.asarray(var.value, dtype=float)

        if values.shape == ():
            return float(values)

        return float(values[q_idx])

    def set_x90_amp(q_idx, amp):
        amp = float(amp)

        if not (-1.0 <= amp <= 1.0):
            raise ValueError(f"x90_amp must be between -1 and 1. got {amp}")

        values = np.asarray(x90_amp.value, dtype=float).copy()

        if values.shape == ():
            x90_amp.value = amp
        else:
            values[q_idx] = amp
            x90_amp.value = values

    anharmonicities = np.zeros(len(q_indices), dtype=float)
    signed_alphas = np.zeros(len(q_indices), dtype=float)
    gf_freqs = np.zeros(len(q_indices), dtype=float)
    ge_freqs = np.zeros(len(q_indices), dtype=float)

    fit_results = []
    program_compileds = []

    pump_is_on = False

    try:
        if pump is not None:
            pump.connect()
            pump.set_on(
                ch=pump_ch,
                freq=pump_freq,
                power=pump_power,
            )
            pump_is_on = True

        for i, q_idx in enumerate(q_indices):
            f_center = float(f_centers[i])

            current_x90_freq = get_var_value(x90_freq, q_idx)
            current_x90_amp = get_var_value(x90_amp, q_idx)

            if x90_spectroscopy_amp is not None:
                spectroscopy_amp = float(x90_spectroscopy_amp[i])
                set_x90_amp(q_idx, spectroscopy_amp)
            else:
                spectroscopy_amp = current_x90_amp

            print(f"Q{q_idx} anharmonicity progress        : {i + 1}/{len(q_indices)}")
            print(f"Q{q_idx} current x90_freq              : {current_x90_freq / GHz:.9f} GHz")
            print(f"Q{q_idx} spectroscopy x90_amp          : {spectroscopy_amp:.6g}")
            print(f"Q{q_idx} sweep center                  : {f_center / GHz:.9f} GHz")
            print(f"Q{q_idx} sweep bandwidth               : {f_range / MHz:.3f} MHz")
            print(f"Q{q_idx} expected anharmonicity         : {expected_anharmonicity / MHz:.3f} MHz")

            freq_scan_values = np.linspace(
                f_center - f_range / 2,
                f_center + f_range / 2,
                n_steps,
            )

            target_qubit = qubits[q_idx]

            qubit_freq_list = qcs.Array(
                f"anharmonicity_freq_list_q{q_idx}",
                value=freq_scan_values,
                dtype=float,
                unit="Hz",
            )

            program = qcs.Program()

            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
                new_layer=True,
            )

            program.add_measurement(
                target_qubit,
                new_layer=True,
            )

            program = program.sweep(
                qubit_freq_list,
                x90_freq[q_idx],
            )

            linker_pass = qcs.LinkerPass(
                *calibration_set.linkers.values()
            )

            program_compiled = linker_pass.apply(program)
            program_compiled.n_shots(n_shots)

            program_compiled = qcs.Executor(backend).execute(program_compiled)

            iq_signal = get_iq_signal(
                program_compiled.get_iq_array(avg=True),
                iq_mode,
            )

            iq_signal = np.asarray(iq_signal).squeeze().reshape(-1)

            if len(iq_signal) != len(freq_scan_values):
                raise RuntimeError(
                    f"Q{q_idx} anharmonicity signal length mismatch: "
                    f"len(iq_signal)={len(iq_signal)}, "
                    f"len(freq_scan_values)={len(freq_scan_values)}"
                )

            fit_result = fit_double_lorentzian_anharmonicity(
                freq_scan_values,
                iq_signal,
                f_ge_ref=current_x90_freq,
                expected_anharmonicity=expected_anharmonicity,
                min_prominence=None,
                fit_on_db=False,
                plot=plot,
            )

            gf_freq = float(fit_result["gf_freq"])
            ge_freq = float(fit_result["ge_freq"])
            anharmonicity = float(fit_result["anharmonicity"])
            signed_alpha = float(fit_result["signed_alpha"])

            gf_freqs[i] = gf_freq
            ge_freqs[i] = ge_freq
            anharmonicities[i] = anharmonicity
            signed_alphas[i] = signed_alpha

            fit_results.append(fit_result)
            program_compileds.append(program_compiled)

            print(f"Q{q_idx} fitted gf/2 freq             : {gf_freq / GHz:.9f} GHz")
            print(f"Q{q_idx} fitted ge freq               : {ge_freq / GHz:.9f} GHz")
            print(f"Q{q_idx} split                        : {(ge_freq - gf_freq) / MHz:.3f} MHz")
            print(f"Q{q_idx} anharmonicity magnitude       : {anharmonicity / MHz:.3f} MHz")
            print(f"Q{q_idx} signed alpha                  : {signed_alpha / MHz:.3f} MHz")

            if clear:
                plt.close("all")
                clear_output(wait=False)

    finally:
        if restore_x90_amp:
            x90_amp.value = original_x90_amp_values

        if pump is not None and pump_is_on:
            pump.off(pump_ch)

    return anharmonicities, fit_results, program_compileds



def _lorentzian_unit(x, f0, kappa):
    return (0.5 * kappa) ** 2 / ((x - f0) ** 2 + (0.5 * kappa) ** 2)


def _double_lorentzian_model(x, y0, A1, f1, k1, A2, f2, k2):
    return (
        y0
        + A1 * _lorentzian_unit(x, f1, k1)
        + A2 * _lorentzian_unit(x, f2, k2)
    )


def fit_double_lorentzian_anharmonicity(
    freqs,
    signal,
    f_ge_ref=None,
    expected_anharmonicity=250e6,
    min_prominence=None,
    fit_on_db=True,
    plot=False,
):
    """
    Strong-drive qubit spectroscopy에서 나타나는
    g-e transition과 two-photon g-f transition을 double Lorentzian으로 fitting.

    낮은 frequency 쪽 center  : gf_half_freq
    높은 frequency 쪽 center  : ge_freq

    anharmonicity = 2 * (ge_freq - gf_half_freq)
    """

    x = np.asarray(freqs, dtype=float).squeeze().reshape(-1)
    raw_y = np.asarray(signal, dtype=float).squeeze().reshape(-1)

    if len(x) != len(raw_y):
        raise RuntimeError(
            f"length mismatch: len(freqs)={len(x)}, len(signal)={len(raw_y)}"
        )

    if len(x) < 7:
        raise RuntimeError("Too few points for double Lorentzian fitting.")

    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(raw_y)):
        raise RuntimeError("Non-finite frequency or signal data.")

    if fit_on_db:
        y = 20 * np.log10(np.maximum(np.abs(raw_y), 1e-15))
    else:
        y = raw_y.copy()

    if np.any(~np.isfinite(y)):
        raise RuntimeError("Non-finite fitting data.")

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    span = x_max - x_min
    df = float(abs(x[1] - x[0]))

    if span <= 0 or df <= 0:
        raise RuntimeError("Invalid frequency axis.")

    edge_n = max(3, len(y) // 10)
    y0_guess = float(np.median(np.r_[y[:edge_n], y[-edge_n:]]))

    dip_strength = y0_guess - float(np.min(y))
    peak_strength = float(np.max(y)) - y0_guess
    is_dip = dip_strength >= peak_strength

    search_y = -y if is_dip else y
    y_ptp = float(np.ptp(y))

    if y_ptp <= 0:
        raise RuntimeError("Signal has no contrast.")

    if min_prominence is None:
        min_prominence = 0.08 * float(np.ptp(search_y))

    peaks, props = find_peaks(
        search_y,
        prominence=min_prominence,
    )

    if len(peaks) >= 2:
        prominences = props["prominences"]
        top2 = peaks[np.argsort(prominences)[-2:]]
        top2 = sorted(top2, key=lambda idx: x[idx])

        f1_guess = float(x[top2[0]])
        f2_guess = float(x[top2[1]])

    else:
        if f_ge_ref is not None:
            f2_guess = float(f_ge_ref)
            f1_guess = float(f_ge_ref - expected_anharmonicity / 2)

            f1_guess = float(np.clip(f1_guess, x_min + 2 * df, x_max - 2 * df))
            f2_guess = float(np.clip(f2_guess, x_min + 2 * df, x_max - 2 * df))

            if abs(f2_guess - f1_guess) < 5 * df:
                f1_guess = x_min + 0.35 * span
                f2_guess = x_min + 0.70 * span
        else:
            f1_guess = x_min + 0.35 * span
            f2_guess = x_min + 0.70 * span

    f1_guess, f2_guess = sorted([f1_guess, f2_guess])

    def y_at(freq):
        return float(y[np.argmin(np.abs(x - freq))])

    A_sign = -1.0 if is_dip else 1.0

    A1_guess = y_at(f1_guess) - y0_guess
    A2_guess = y_at(f2_guess) - y0_guess

    if A_sign * A1_guess <= 0:
        A1_guess = A_sign * max(y_ptp / 3, 1e-9)

    if A_sign * A2_guess <= 0:
        A2_guess = A_sign * max(y_ptp / 3, 1e-9)

    kappa_guess = max(span / 30, df)

    p0 = [
        y0_guess,
        A1_guess,
        f1_guess,
        kappa_guess,
        A2_guess,
        f2_guess,
        kappa_guess,
    ]

    if is_dip:
        amp_lower = -np.inf
        amp_upper = 0.0
    else:
        amp_lower = 0.0
        amp_upper = np.inf

    lower = [
        float(np.min(y) - 2 * y_ptp),
        amp_lower,
        x_min,
        df,
        amp_lower,
        x_min,
        df,
    ]

    upper = [
        float(np.max(y) + 2 * y_ptp),
        amp_upper,
        x_max,
        span,
        amp_upper,
        x_max,
        span,
    ]

    try:
        popt, pcov = curve_fit(
            _double_lorentzian_model,
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
    except Exception as e:
        raise RuntimeError(f"Double Lorentzian fitting failed: {e}")

    y0, A1, f1, k1, A2, f2, k2 = popt

    centers = [
        {
            "f": float(f1),
            "A": float(A1),
            "kappa": float(k1),
        },
        {
            "f": float(f2),
            "A": float(A2),
            "kappa": float(k2),
        },
    ]

    centers = sorted(centers, key=lambda item: item["f"])

    gf_freq = centers[0]["f"]
    ge_freq = centers[1]["f"]
    split = ge_freq - gf_freq
    anharmonicity = 2 * split
    signed_alpha = -anharmonicity

    fit_y = _double_lorentzian_model(x, *popt)

    if plot:
        plt.figure(figsize=(7, 4))
        plt.plot(x / GHz, y, "o", ms=4, label="data")
        plt.plot(x / GHz, fit_y, "-", label="double Lorentzian fit")
        plt.axvline(gf_freq / GHz, linestyle="--", label="gf/2 transition")
        plt.axvline(ge_freq / GHz, linestyle="--", label="ge transition")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Signal (dB)" if fit_on_db else "Signal")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "status": "ok",
        "gf_freq": gf_freq,
        "ge_freq": ge_freq,
        "split": split,
        "anharmonicity": anharmonicity,
        "signed_alpha": signed_alpha,
        "popt": popt,
        "pcov": pcov,
        "fit_x": x,
        "fit_y": fit_y,
        "y": y,
        "is_dip": is_dip,
    }

def measure_t1s(
    t_ranges,
    backend,
    calibration_set,
    qubits,
    pump_ch,
    pump_freq,
    pump_power,
    n_steps,
    n_shots,
    q_indices=None,
    iq_mode="abs",
    clear=False,
    pump=None,
):
    
    """
    T1 measurement.

    |g> --x90--x90--> |e>
    wait delay
    measure

    delay를 sweep하고 t1_fit으로 T1 추정.
    """

    x90_freq = calibration_set.variables.x90_freq
    x90_amp = calibration_set.variables.x90_amp
     
    t_ranges = np.asarray(t_ranges, dtype=float)

    if t_ranges.ndim == 1:
        t_ranges = t_ranges.reshape(1, -1)

    if q_indices is None:
        q_indices = range(len(t_ranges))
    q_indices = list(q_indices)


    if t_ranges.shape[0] == 1 and len(q_indices) > 1:
        t_ranges = np.repeat(t_ranges, len(q_indices), axis=0)

    if t_ranges.shape[0] != len(q_indices):
        raise ValueError(
            f"t_ranges must have 1 row or len(q_indices) rows. "
            f"got t_ranges.shape[0]={t_ranges.shape[0]}, "
            f"len(q_indices)={len(q_indices)}"
        )


    
    t1s = np.zeros(len(q_indices), dtype=float)
    program_compileds = []

    pump_is_on = False


    try:
        if pump is not None:
            pump.connect()
            pump.set_on(
                ch=pump_ch,
                freq=pump_freq,
                power=pump_power,
            )
            pump_is_on = True

        for i, q_idx in enumerate(q_indices):
            t_range = t_ranges[i]

            time_scan_values = np.linspace(
                t_range[0],
                t_range[-1],
                n_steps,
            )

            target_qubit = qubits[q_idx]

            x90_freq_values = np.asarray(x90_freq.value, dtype=float)
            x90_amp_values = np.asarray(x90_amp.value, dtype=float)

            if x90_freq_values.shape == ():
                current_x90_freq = float(x90_freq_values)
            else:
                current_x90_freq = float(x90_freq_values[q_idx])

            if x90_amp_values.shape == ():
                current_x90_amp = float(x90_amp_values)
            else:
                current_x90_amp = float(x90_amp_values[q_idx])

            print(f"Q{q_idx} T1 progress          : {i + 1}/{len(q_indices)}")
            print(f"Q{q_idx} current x90_freq     : {current_x90_freq / GHz:.9f} GHz")
            print(f"Q{q_idx} current x90_amp      : {current_x90_amp:.6g}")
            print(f"Q{q_idx} T1 delay range       : {time_scan_values[0] / us:.3f} ~ {time_scan_values[-1] / us:.3f} us")
            print(f"Q{q_idx} T1 delay points      : {n_steps}")

            t1_delay = qcs.Scalar(
                f"t1_delay_q{q_idx}",
                value=float(time_scan_values[0]),
                dtype=float,
                unit="s",
            )

            t1_delay_list = qcs.Array(
                f"t1_delay_list_q{q_idx}",
                value=time_scan_values,
                dtype=float,
                unit="s",
            )

            program = qcs.Program()

            # |e> preparation: x90 + x90 = X
            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
                new_layer=True,
            )

            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
                new_layer=True,
            )

            # variable delay 후 measurement
            program.add_measurement(
                target_qubit,
                pre_delay=t1_delay,
                new_layer=True,
            )

            program = program.sweep(
                t1_delay_list,
                t1_delay,
            )

            linker_pass = qcs.LinkerPass(
                *calibration_set.linkers.values()
            )

            program_compiled = linker_pass.apply(program)
            program_compiled.n_shots(n_shots)

            program_compiled = qcs.Executor(backend).execute(program_compiled)

            iq_signal = get_iq_signal(
                program_compiled.get_iq_array(avg=True),
                iq_mode,
            )

            iq_signal = np.asarray(iq_signal).squeeze().reshape(-1)

            if len(iq_signal) != len(time_scan_values):
                raise RuntimeError(
                    f"Q{q_idx} T1 signal length mismatch: "
                    f"len(iq_signal)={len(iq_signal)}, "
                    f"len(time_scan_values)={len(time_scan_values)}"
                )

            t1 = t1_fit(
                time_scan_values,
                iq_signal,
            )

            t1s[i] = float(t1)
            program_compileds.append(program_compiled)

            print(f"Q{q_idx} fitted T1           : {t1s[i] / us:.3f} us")

            if clear:
                plt.close("all")
                clear_output(wait=False)

    finally:
        if pump is not None and pump_is_on:
            pump.off(pump_ch)

    return t1s, program_compileds


def measure_ramseys(
    t_ranges,
    backend,
    calibration_set,
    qubits,
    pump_ch,
    pump_freq,
    pump_power,
    n_steps,
    n_shots,
    q_indices=None,
    iq_mode="phase",
    clear=False,
    pump=None,
):
    
    x90_freq = calibration_set.variables.x90_freq
    x90_amp = calibration_set.variables.x90_amp

    t_ranges = np.asarray(t_ranges, dtype=float)

    if t_ranges.ndim == 1:
        t_ranges = t_ranges.reshape(1, -1)

    warnings.filterwarnings("ignore", message=r".*was rounded.*")

    if q_indices is None:
        q_indices = range(len(qubits.labels))
    q_indices = list(q_indices)

    if t_ranges.shape[0] == 1 and len(q_indices) > 1:
        t_ranges = np.repeat(t_ranges, len(q_indices), axis=0)

    if t_ranges.shape[0] != len(q_indices):
        raise ValueError(
            f"t_ranges must have 1 row or len(q_indices) rows. "
            f"got t_ranges.shape[0]={t_ranges.shape[0]}, "
            f"len(q_indices)={len(q_indices)}"
        )

    iq_datas = np.zeros((len(q_indices), n_steps), dtype=float)
    time_axes = []
    program_compileds = []

    pump_is_on = False

    try:
        if pump is not None:
            pump.connect()
            pump.set_on(
                ch=pump_ch,
                freq=pump_freq,
                power=pump_power,
            )
            pump_is_on = True

        for i, q_idx in enumerate(q_indices):
            t_range = t_ranges[i]

            time_scan_values = np.linspace(
                t_range[0],
                t_range[-1],
                n_steps,
            )

            target_qubit = qubits[q_idx]

            x90_freq_values = np.asarray(x90_freq.value, dtype=float)
            x90_amp_values = np.asarray(x90_amp.value, dtype=float)

            if x90_freq_values.shape == ():
                current_x90_freq = float(x90_freq_values)
            else:
                current_x90_freq = float(x90_freq_values[q_idx])

            if x90_amp_values.shape == ():
                current_x90_amp = float(x90_amp_values)
            else:
                current_x90_amp = float(x90_amp_values[q_idx])

            print(f"Q{q_idx} Ramsey progress       : {i + 1}/{len(q_indices)}")
            print(f"Q{q_idx} current x90_freq      : {current_x90_freq / GHz:.9f} GHz")
            print(f"Q{q_idx} current x90_amp       : {current_x90_amp:.6g}")
            print(f"Q{q_idx} Ramsey delay range    : {time_scan_values[0] / us:.3f} ~ {time_scan_values[-1] / us:.3f} us")
            print(f"Q{q_idx} Ramsey delay points   : {n_steps}")

            ramsey_delay = qcs.Scalar(
                f"ramsey_delay_q{q_idx}",
                value=float(time_scan_values[0]),
                dtype=float,
                unit="s",
            )

            ramsey_delay_list = qcs.Array(
                f"ramsey_delay_list_q{q_idx}",
                value=time_scan_values,
                dtype=float,
                unit="s",
            )

            program = qcs.Program()

            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
                new_layer=True,
            )

            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
                pre_delay=ramsey_delay,
                new_layer=True,
            )

            program.add_measurement(
                target_qubit,
                new_layer=True,
            )

            program = program.sweep(
                ramsey_delay_list,
                ramsey_delay,
            )

            linker_pass = qcs.LinkerPass(
                *calibration_set.linkers.values()
            )

            program_compiled = linker_pass.apply(program)
            program_compiled.n_shots(n_shots)

            program_compiled = qcs.Executor(backend).execute(program_compiled)

            iq_data = get_iq_signal(
                program_compiled.get_iq_array(avg=True),
                iq_mode,
            )

            iq_data = np.asarray(iq_data).squeeze().reshape(-1)

            if len(iq_data) != len(time_scan_values):
                raise RuntimeError(
                    f"Q{q_idx} Ramsey signal length mismatch: "
                    f"len(iq_data)={len(iq_data)}, "
                    f"len(time_scan_values)={len(time_scan_values)}"
                )

            iq_datas[i, :] = iq_data
            time_axes.append(time_scan_values)
            program_compileds.append(program_compiled)

            if clear:
                plt.close("all")
                clear_output(wait=False)

    finally:
        if pump is not None and pump_is_on:
            pump.off(pump_ch)

    return {
        "iq_datas": iq_datas,
        "time_axes": time_axes,
        "q_indices": q_indices,
        "programs": program_compileds,
    }


def fit_ramseys(
    ramsey_result,
    t2_guess,
    delta_guess,
    plot="all",
):
    """
    measure_ramseys()의 반환값을 받아 Ramsey fitting 수행.

    """

    iq_datas = np.asarray(ramsey_result["iq_datas"], dtype=float)
    time_axes = ramsey_result["time_axes"]
    q_indices = list(ramsey_result["q_indices"])

    n_qubits = len(q_indices)

    t2_guess = np.asarray(t2_guess, dtype=float)
    delta_guess = np.asarray(delta_guess, dtype=float)

    if t2_guess.shape == ():
        t2_guess = np.repeat(float(t2_guess), n_qubits)

    if delta_guess.shape == ():
        delta_guess = np.repeat(float(delta_guess), n_qubits)

    if len(t2_guess) == 1 and n_qubits > 1:
        t2_guess = np.repeat(t2_guess[0], n_qubits)

    if len(delta_guess) == 1 and n_qubits > 1:
        delta_guess = np.repeat(delta_guess[0], n_qubits)

    if len(t2_guess) != n_qubits:
        raise ValueError(
            f"len(t2_guess) must be 1 or len(q_indices). "
            f"got len(t2_guess)={len(t2_guess)}, len(q_indices)={n_qubits}"
        )

    if len(delta_guess) != n_qubits:
        raise ValueError(
            f"len(delta_guess) must be 1 or len(q_indices). "
            f"got len(delta_guess)={len(delta_guess)}, len(q_indices)={n_qubits}"
        )

    if plot is None:
        plot_indices = []
    elif plot == "all":
        plot_indices = range(n_qubits)
    else:
        plot_indices = list(plot)

    t2s = np.zeros(n_qubits, dtype=float)
    detunings = np.zeros(n_qubits, dtype=float)

    for i, q_idx in enumerate(q_indices):
        time_scan_values = np.asarray(time_axes[i], dtype=float).squeeze()
        iq_data = np.asarray(iq_datas[i], dtype=float).squeeze()

        do_plot = i in plot_indices

        t2, detuning = t2_fit(
            time_scan_values,
            iq_data,
            t2_guess[i],
            delta_guess[i],
            plot=do_plot,
        )

        t2s[i] = float(t2)
        detunings[i] = float(detuning)

        print(f"Q{q_idx} fitted T2*       : {t2s[i] / us:.3f} us")
        print(f"Q{q_idx} fitted detuning  : {detunings[i] / MHz:.3f} MHz")

    return {
        "t2s": t2s,
        "detunings": detunings,
        "q_indices": q_indices,
    }


def measure_ef_frequencies(
    backend,
    calibration_set,
    qubits,
    pump_ch,
    pump_freq,
    pump_power,
    n_steps,
    n_shots,
    f0_gf_half=None,
    f_ef_guess=None,
    ef_dur_value=50 * ns,
    ef_amp_val=None,
    sweep_range=500 * MHz,
    q_indices=None,
    iq_mode="phase",
    clear=False,
    pump=None,
):
    """
    e-f transition frequency measurement.

    |g> --x90--x90--> |e>
    e-f pulse frequency sweep
    measure

    """

    x90_freq = calibration_set.variables.x90_freq
    x90_amp = calibration_set.variables.x90_amp

    warnings.filterwarnings("ignore", message=r".*was rounded.*")

    def get_input_length(x):
        if x is None:
            return None

        x = np.asarray(x, dtype=float)

        if x.shape == ():
            return 1

        return len(x)

    def get_input_value(x, i):
        if x is None:
            return None

        x = np.asarray(x, dtype=float)

        if x.shape == ():
            return float(x)

        return float(x[i])

    def get_var_value(var, q_idx):
        values = np.asarray(var.value, dtype=float)

        if values.shape == ():
            return float(values)

        return float(values[q_idx])

    if q_indices is None:
        n_inputs = get_input_length(f_ef_guess)

        if n_inputs is None:
            n_inputs = get_input_length(f0_gf_half)

        if n_inputs is None:
            q_indices = range(len(qubits.labels))
        else:
            q_indices = range(n_inputs)

    q_indices = list(q_indices)

    if f0_gf_half is None and f_ef_guess is None:
        raise ValueError("Either f0_gf_half or f_ef_guess must be provided.")

    ef_freqs = np.zeros(len(q_indices), dtype=float)
    program_compileds = []
    fit_results = []

    xy_channels = qcs.Channels(
        qubits.labels,
        "xy_pulse",
        absolute_phase=False,
    )

    pump_is_on = False

    try:
        if pump is not None:
            pump.connect()
            pump.set_on(
                ch=pump_ch,
                freq=pump_freq,
                power=pump_power,
            )
            pump_is_on = True

        for i, q_idx in enumerate(q_indices):
            target_qubit = qubits[q_idx]

            f_ge = get_var_value(x90_freq, q_idx)
            x90_amp_i = get_var_value(x90_amp, q_idx)

            f0_gf_half_i = get_input_value(f0_gf_half, i)
            f_ef_guess_i = get_input_value(f_ef_guess, i)

            if f0_gf_half_i is None and f_ef_guess_i is None:
                raise ValueError(
                    f"Q{q_idx}: Either f0_gf_half or f_ef_guess must be provided."
                )

            if f_ef_guess_i is None:
                f_ef_guess_i = 2 * f0_gf_half_i - f_ge

            if f0_gf_half_i is None:
                f0_gf_half_i = (f_ge + f_ef_guess_i) / 2

            if ef_amp_val is None:
                ef_amp_i = np.sqrt(2) * x90_amp_i
            else:
                ef_amp_i = get_input_value(ef_amp_val, i)

            if not (-1.0 <= ef_amp_i <= 1.0):
                raise ValueError(
                    f"Q{q_idx} ef_amp must be between -1 and 1. got {ef_amp_i}"
                )

            print(f"Q{q_idx} ef_freq progress       : {i + 1}/{len(q_indices)}")
            print(f"Q{q_idx} f_ge                   : {f_ge / GHz:.9f} GHz")
            print(f"Q{q_idx} f0_gf_half             : {f0_gf_half_i / GHz:.9f} GHz")
            print(f"Q{q_idx} f_ef_guess             : {f_ef_guess_i / GHz:.9f} GHz")
            print(f"Q{q_idx} ef_amp                 : {ef_amp_i:.6g}")
            print(f"Q{q_idx} ef sweep bandwidth     : {sweep_range / MHz:.3f} MHz")

            ef_freq_scan_values = np.linspace(
                f_ef_guess_i - sweep_range / 2,
                f_ef_guess_i + sweep_range / 2,
                n_steps,
            )

            ef_freq = qcs.Scalar(
                f"ef_freq_q{q_idx}",
                value=f_ef_guess_i,
                dtype=float,
                unit="Hz",
            )

            ef_dur = qcs.Scalar(
                f"ef_dur_q{q_idx}",
                value=ef_dur_value,
                dtype=float,
                unit="s",
            )

            ef_amp = qcs.Scalar(
                f"ef_amp_q{q_idx}",
                value=ef_amp_i,
                dtype=float,
                unit="",
            )

            ef_freq_list = qcs.Array(
                f"ef_freq_list_q{q_idx}",
                value=ef_freq_scan_values,
                dtype=float,
                unit="Hz",
            )

            ef_pulse = qcs.RFWaveform(
                ef_dur,
                qcs.GaussianEnvelope(),
                ef_amp,
                rf_frequency=ef_freq,
            )

            program = qcs.Program()

            # Prepare |e> with x90 + x90
            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
                new_layer=True,
            )

            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
                new_layer=True,
            )

            # Sweep e-f pulse frequency
            program.add_waveform(
                ef_pulse,
                xy_channels[q_idx],
                new_layer=True,
            )

            program.add_measurement(
                target_qubit,
                new_layer=True,
            )

            program = program.sweep(
                ef_freq_list,
                ef_freq,
            )

            linker_pass = qcs.LinkerPass(
                *calibration_set.linkers.values()
            )

            program_compiled = linker_pass.apply(program)
            program_compiled.n_shots(n_shots)

            program_compiled = qcs.Executor(backend).execute(program_compiled)

            iq_signal = get_iq_signal(
                program_compiled.get_iq_array(avg=True),
                iq_mode,
            )

            iq_signal = np.asarray(iq_signal).squeeze().reshape(-1)

            if len(iq_signal) != len(ef_freq_scan_values):
                raise RuntimeError(
                    f"Q{q_idx} EF signal length mismatch: "
                    f"len(iq_signal)={len(iq_signal)}, "
                    f"len(ef_freq_scan_values)={len(ef_freq_scan_values)}"
                )

            fit_result = fit_lorentzian(
                ef_freq_scan_values,
                iq_signal,
                plot=False,
            )
            fit_results.append(fit_result)

            ef_freqs[i] = float(fit_result["f0"])
            program_compileds.append(program_compiled)

            print(f"Q{q_idx} fitted f_ef           : {ef_freqs[i] / GHz:.9f} GHz")
            print(f"Q{q_idx} anharmonicity estimate: {(f_ge - ef_freqs[i]) / MHz:.3f} MHz")

            if clear:
                plt.close("all")
                clear_output(wait=False)

    finally:
        if pump is not None and pump_is_on:
            pump.off(pump_ch)

    return ef_freqs, fit_results, program_compileds


def measure_ef_rabi(
    ef_freqs,
    backend,
    calibration_set,
    qubits,
    pump_ch,
    pump_freq,
    pump_power,
    n_steps,
    n_shots,
    amp_range,
    ef_dur_value=50*ns,
    q_indices=None,
    iq_mode="phase"
):
    """
    e-f Rabi measurement.

    |g> --x90--x90--> |e>
    e-f pulse amplitude sweep
    measure

    """
    warnings.filterwarnings("ignore", message=r".*was rounded.*")

    x90_freq = calibration_set.variables.x90_freq
    x90_amp = calibration_set.variables.x90_amp

    ef_freqs = np.asarray(ef_freqs, dtype=float)

    if ef_freqs.shape == ():
        ef_freqs = np.array([float(ef_freqs)], dtype=float)

    if q_indices is None:
        if len(ef_freqs) == 1:
            q_indices = range(len(qubits.labels))
        else:
            q_indices = range(len(ef_freqs))

    q_indices = list(q_indices)

    if len(ef_freqs) == 1 and len(q_indices) > 1:
        ef_freqs = np.repeat(ef_freqs[0], len(q_indices))

    if len(ef_freqs) != len(q_indices):
        raise ValueError(
            f"len(ef_freqs) must be 1 or len(q_indices). "
            f"got len(ef_freqs)={len(ef_freqs)}, "
            f"len(q_indices)={len(q_indices)}"
        )

    amp_ranges = np.asarray(amp_range, dtype=float)

    if amp_ranges.ndim == 1:
        amp_ranges = amp_ranges.reshape(1, -1)

    if amp_ranges.shape[0] == 1 and len(q_indices) > 1:
        amp_ranges = np.repeat(amp_ranges, len(q_indices), axis=0)

    if amp_ranges.shape[0] != len(q_indices):
        raise ValueError(
            f"amp_range must have 1 row or len(q_indices) rows. "
            f"got amp_range.shape[0]={amp_ranges.shape[0]}, "
            f"len(q_indices)={len(q_indices)}"
        )

    def get_var_value(var, q_idx):
        values = np.asarray(var.value, dtype=float)

        if values.shape == ():
            return float(values)

        return float(values[q_idx])

    ef_amps = np.zeros(len(q_indices), dtype=float)
    program_compileds = []

    xy_channels = qcs.Channels(
        qubits.labels,
        "xy_pulse",
        absolute_phase=False,
    )

    pump_is_on = False

    try:
        if pump is not None:
            pump.connect()
            pump.set_on(
                ch=pump_ch,
                freq=pump_freq,
                power=pump_power,
            )
            pump_is_on = True

        for i, q_idx in enumerate(q_indices):
            target_qubit = qubits[q_idx]

            f_ef = float(ef_freqs[i])
            amp_range_i = amp_ranges[i]

            ef_amp_scan_values = np.linspace(
                amp_range_i[0],
                amp_range_i[-1],
                n_steps,
            )

            if np.any(np.abs(ef_amp_scan_values) > 1.0):
                raise ValueError(
                    f"Q{q_idx} ef_amp sweep contains value outside [-1, 1]. "
                    f"amp range = [{ef_amp_scan_values.min()}, "
                    f"{ef_amp_scan_values.max()}]"
                )

            current_x90_freq = get_var_value(x90_freq, q_idx)
            current_x90_amp = get_var_value(x90_amp, q_idx)

            print(f"Q{q_idx} ef_rabi progress       : {i + 1}/{len(q_indices)}")
            print(f"Q{q_idx} current x90_freq       : {current_x90_freq / GHz:.9f} GHz")
            print(f"Q{q_idx} current x90_amp        : {current_x90_amp:.6g}")
            print(f"Q{q_idx} f_ef                   : {f_ef / GHz:.9f} GHz")
            print(f"Q{q_idx} ef_amp sweep range     : {ef_amp_scan_values[0]:.6g} ~ {ef_amp_scan_values[-1]:.6g}")
            print(f"Q{q_idx} ef_amp sweep points    : {n_steps}")

            ef_freq = qcs.Scalar(
                f"ef_freq_q{q_idx}",
                value=f_ef,
                dtype=float,
                unit="Hz",
            )

            ef_amp = qcs.Scalar(
                f"ef_amp_q{q_idx}",
                value=float(ef_amp_scan_values[0]),
                dtype=float,
                unit="",
            )

            ef_dur = qcs.Scalar(
                f"ef_dur_q{q_idx}",
                value=ef_dur_value,
                dtype=float,
                unit="s",
            )

            ef_amp_list = qcs.Array(
                f"ef_amp_list_q{q_idx}",
                value=ef_amp_scan_values,
                dtype=float,
                unit="",
            )

            ef_pulse = qcs.RFWaveform(
                ef_dur,
                qcs.GaussianEnvelope(),
                ef_amp,
                rf_frequency=ef_freq,
            )

            program = qcs.Program()

            # Prepare |e> with x90 + x90
            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
                new_layer=True,
            )

            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
                new_layer=True,
            )

            # Sweep e-f pulse amplitude
            program.add_waveform(
                ef_pulse,
                xy_channels[q_idx],
                new_layer=True,
            )

            program.add_measurement(
                target_qubit,
                new_layer=True,
            )

            program = program.sweep(
                ef_amp_list,
                ef_amp,
            )

            linker_pass = qcs.LinkerPass(
                *calibration_set.linkers.values()
            )

            program_compiled = linker_pass.apply(program)
            program_compiled.n_shots(n_shots)

            program_compiled = qcs.Executor(backend).execute(program_compiled)

            iq_signal = get_iq_signal(
                program_compiled.get_iq_array(avg=True),
                iq_mode,
            )

            iq_signal = np.asarray(iq_signal).squeeze().reshape(-1)

            if len(iq_signal) != len(ef_amp_scan_values):
                raise RuntimeError(
                    f"Q{q_idx} EF Rabi signal length mismatch: "
                    f"len(iq_signal)={len(iq_signal)}, "
                    f"len(ef_amp_scan_values)={len(ef_amp_scan_values)}"
                )

            _, ef_amp_fit = amplitude_rabi_fit(
                ef_amp_scan_values,
                iq_signal,
            )

            ef_amps[i] = float(ef_amp_fit)
            program_compileds.append(program_compiled)

            print(f"Q{q_idx} fitted ef_amp          : {ef_amps[i]:.9g}")

            if clear:
                plt.close("all")
                clear_output(wait=False)

    finally:
        if pump is not None and pump_is_on:
            pump.off(pump_ch)

    return ef_amps, program_compileds



def measure_echo_t2s(
    t_ranges,
    backend,
    calibration_set,
    qubits,
    pump_ch,
    pump_freq,
    pump_power,
    n_steps,
    n_shots,
    t2_guess=100,
    q_indices=None,
    iq_mode="phase",
    clear=False,
    pump=None,
):
    """
    Echo T2 measurement.

    Sequence:
        x90
        wait tau / 2
        x90 + x90   # x180 pulse
        wait tau / 2
        x90
        measure

    tau를 sweep하고 echo_fit으로 T2_echo 추정.
    """

    x90_freq = calibration_set.variables.x90_freq
    x90_amp = calibration_set.variables.x90_amp

    t_ranges = np.asarray(t_ranges, dtype=float)

    if t_ranges.ndim == 1:
        t_ranges = t_ranges.reshape(1, -1)

    warnings.filterwarnings("ignore", message=r".*was rounded.*")

    if q_indices is None:
        q_indices = range(len(qubits.labels))
    q_indices = list(q_indices)

    if t_ranges.shape[0] == 1 and len(q_indices) > 1:
        t_ranges = np.repeat(t_ranges, len(q_indices), axis=0)

    if t_ranges.shape[0] != len(q_indices):
        raise ValueError(
            f"t_ranges must have 1 row or len(q_indices) rows. "
            f"got t_ranges.shape[0]={t_ranges.shape[0]}, "
            f"len(q_indices)={len(q_indices)}"
        )

    t2_guess = np.asarray(t2_guess, dtype=float)

    if t2_guess.shape == ():
        t2_guess = np.repeat(float(t2_guess), len(q_indices))

    if len(t2_guess) == 1 and len(q_indices) > 1:
        t2_guess = np.repeat(t2_guess[0], len(q_indices))

    if len(t2_guess) != len(q_indices):
        raise ValueError(
            f"len(t2_guess) must be 1 or len(q_indices). "
            f"got len(t2_guess)={len(t2_guess)}, len(q_indices)={len(q_indices)}"
        )

    t2_echoes = np.zeros(len(q_indices), dtype=float)
    iq_datas = np.zeros((len(q_indices), n_steps), dtype=float)
    time_axes = []
    program_compileds = []

    pump_is_on = False

    try:
        if pump is not None:
            pump.connect()
            pump.set_on(
                ch=pump_ch,
                freq=pump_freq,
                power=pump_power,
            )
            pump_is_on = True

        for i, q_idx in enumerate(q_indices):
            t_range = t_ranges[i]

            time_scan_values = np.linspace(
                t_range[0],
                t_range[-1],
                n_steps,
            )

            half_time_scan_values = time_scan_values / 2

            target_qubit = qubits[q_idx]

            x90_freq_values = np.asarray(x90_freq.value, dtype=float)
            x90_amp_values = np.asarray(x90_amp.value, dtype=float)

            if x90_freq_values.shape == ():
                current_x90_freq = float(x90_freq_values)
            else:
                current_x90_freq = float(x90_freq_values[q_idx])

            if x90_amp_values.shape == ():
                current_x90_amp = float(x90_amp_values)
            else:
                current_x90_amp = float(x90_amp_values[q_idx])

            print(f"Q{q_idx} Echo T2 progress       : {i + 1}/{len(q_indices)}")
            print(f"Q{q_idx} current x90_freq       : {current_x90_freq / GHz:.9f} GHz")
            print(f"Q{q_idx} current x90_amp        : {current_x90_amp:.6g}")
            print(f"Q{q_idx} Echo total delay range : {time_scan_values[0] / us:.3f} ~ {time_scan_values[-1] / us:.3f} us")
            print(f"Q{q_idx} Echo half delay range  : {half_time_scan_values[0] / us:.3f} ~ {half_time_scan_values[-1] / us:.3f} us")
            print(f"Q{q_idx} Echo delay points      : {n_steps}")

            echo_half_delay = qcs.Scalar(
                f"echo_half_delay_q{q_idx}",
                value=float(half_time_scan_values[0]),
                dtype=float,
                unit="s",
            )

            echo_half_delay_list = qcs.Array(
                f"echo_half_delay_list_q{q_idx}",
                value=half_time_scan_values,
                dtype=float,
                unit="s",
            )

            program = qcs.Program()

            # First pi/2 pulse
            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
                new_layer=True,
            )

            # Middle pi pulse = x90 + x90
            # 앞에 echo_half_delay를 넣어서:
            # x90 -- tau/2 -- x90 -- x90
            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
                pre_delay=echo_half_delay,
                new_layer=True,
            )

            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
                new_layer=True,
            )

            # Final pi/2 pulse
            # 앞에 echo_half_delay를 한 번 더 넣어서:
            # x90 -- tau/2 -- x180 -- tau/2 -- x90
            program.add_gate(
                qcs.GATES.x90,
                target_qubit,
                pre_delay=echo_half_delay,
                new_layer=True,
            )

            program.add_measurement(
                target_qubit,
                new_layer=True,
            )

            program = program.sweep(
                echo_half_delay_list,
                echo_half_delay,
            )

            linker_pass = qcs.LinkerPass(
                *calibration_set.linkers.values()
            )

            program_compiled = linker_pass.apply(program)
            program_compiled.n_shots(n_shots)

            program_compiled = qcs.Executor(backend).execute(program_compiled)

            iq_data = get_iq_signal(
                program_compiled.get_iq_array(avg=True),
                iq_mode,
            )

            iq_data = np.asarray(iq_data).squeeze().reshape(-1)

            if len(iq_data) != len(time_scan_values):
                raise RuntimeError(
                    f"Q{q_idx} Echo T2 signal length mismatch: "
                    f"len(iq_data)={len(iq_data)}, "
                    f"len(time_scan_values)={len(time_scan_values)}"
                )

            t2_echo = echo_fit(
                time_scan_values,
                iq_data,
                t2_init=t2_guess[i],
            )

            t2_echoes[i] = float(t2_echo)
            iq_datas[i, :] = iq_data
            time_axes.append(time_scan_values)
            program_compileds.append(program_compiled)

            print(f"Q{q_idx} fitted Echo T2        : {t2_echoes[i] / us:.3f} us")

            if clear:
                plt.close("all")
                clear_output(wait=False)

    finally:
        if pump is not None and pump_is_on:
            pump.off(pump_ch)

    return {
        "t2_echoes": t2_echoes,
        "iq_datas": iq_datas,
        "time_axes": time_axes,
        "q_indices": q_indices,
        "programs": program_compileds,
    }