import json
import re
import numpy as np
import pandas as pd

import keysight.qcs as qcs
from keysight.qcs.channels import (
    ConstantEnvelope,
    GaussianEnvelope,
    PhaseIncrement,
    RFWaveform,
    SineEnvelope,
)

from keysight.qcs.programs import CalibrationSet
from keysight.qcs.quantum import (
    GATES,
    PAULIS,
    ParameterizedGate,
    ParametricGate,
)

from keysight.qcs.variables import Array, Scalar


def set_value_at(var, value, index):
    v = var.value.copy()

    v[index] = value

    var.value = v


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def validate_json(variables_from_json, qubits, variable_names):
    missing_qudits = []
    missing_params = {}

    for label in qubits.labels:
        key = f"q_{label}"

        if key not in variables_from_json:
            missing_qudits.append(key)
            continue

        for name in variable_names:
            if name not in variables_from_json[key]:
                missing_params.setdefault(key, []).append(name)

    if missing_qudits or missing_params:
        message = "Invalid variable format from json."

        if missing_qudits:
            message += f"\nMissing qudit keys: {missing_qudits}"

        if missing_params:
            message += f"\nMissing parameters: {missing_params}"

        raise KeyError(message)
    

def json_to_dataframe(json_data):
    rows = []

    for qudit_key, params in json_data.items():
        row = {}

        # qudit index 추출: "qudits_0" -> 0
        row["q"] = int(qudit_key.split("_")[-1])

        for name, value in params.items():
            if name == "classification_refs":
                real = np.asarray(value["real"], dtype=float)
                imag = np.asarray(value["imag"], dtype=float)

                row["classification_refs"] = real + 1j * imag
            else:
                row[name] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    # qudit 순서대로 정렬
    df = df.sort_values("q").reset_index(drop=True)

    return df

  

def is_missing_value(value):
    """
    Return True only for scalar NaN/None values.

    Arrays, lists, and tuples are not treated as missing here,
    because pd.isna(array) returns an array of booleans.
    """
    if value is None:
        return True

    if isinstance(value, (list, tuple, np.ndarray)):
        return False

    return pd.isna(value)


def dataframe_to_json(df, filename=None):
    """
    Convert a calibration DataFrame to JSON-like dictionary.

    If filename is given, save the JSON dictionary to that file.

    Parameters
    ----------
    df : pd.DataFrame
        Calibration DataFrame.
    filename : str, optional
        Output JSON filename. If None, the function only returns the dictionary.

    Returns
    -------
    dict
        JSON-compatible calibration dictionary.
    """
    json_data = {}

    for _, row in df.iterrows():
        qudit = int(row["q"])
        qudit_key = f"q_{qudit}"

        params = {}
        real_refs = {}
        imag_refs = {}

        for col in df.columns:
            if col == "q":
                continue

            value = row[col]

            if is_missing_value(value):
                continue

            real_match = re.match(r"classification_ref_(\d+)_real$", col)
            imag_match = re.match(r"classification_ref_(\d+)_imag$", col)

            if real_match:
                idx = int(real_match.group(1))
                real_refs[idx] = float(value)
                continue

            if imag_match:
                idx = int(imag_match.group(1))
                imag_refs[idx] = float(value)
                continue

            # Skip convenience complex columns such as classification_ref_0
            if re.match(r"classification_ref_\d+$", col):
                continue

            # Handle non-flattened classification_refs column
            if col == "classification_refs":
                refs = np.asarray(value, dtype=complex)

                params["classification_refs"] = {
                    "real": np.real(refs).astype(float).tolist(),
                    "imag": np.imag(refs).astype(float).tolist(),
                }
                continue

            if isinstance(value, np.generic):
                value = value.item()

            params[col] = value

        # Reconstruct classification_refs from flattened real/imag columns
        if "classification_refs" not in params and (real_refs or imag_refs):
            ref_indices = sorted(set(real_refs.keys()) | set(imag_refs.keys()))

            params["classification_refs"] = {
                "real": [float(real_refs.get(i, 0.0)) for i in ref_indices],
                "imag": [float(imag_refs.get(i, 0.0)) for i in ref_indices],
            }

        json_data[qudit_key] = params

    if filename is not None:
        with open(filename, "w") as f:
            json.dump(json_data, f, indent=2)

    return json_data


def make_qcs_var(variables_from_json, qubits):
    """
    Convert JSON calibration parameters into a list of QCS Array variables.

    Returns
    -------
    list
        List of QCS Array variables, including classification_refs.
    """

    first_key = next(iter(variables_from_json))
    variable_names=[name for name in variables_from_json[first_key].keys() if name != 'classification_refs']

    qubit_labels = qubits.labels

    if not isinstance(qubit_labels, tuple):
        raise TypeError(
            f"qubits.labels must be a tuple, but got {type(qubit_labels)}."
        )

    validate_json(
        variables_from_json=variables_from_json,
        qubits=qubits,
        variable_names=variable_names,
    )

    qcs_vars = []

    # Regular scalar parameters
    for name in variable_names:
        values = np.asarray(
            [
                variables_from_json[f"q_{label}"][name]
                for label in qubit_labels
            ],
            dtype=float,
        )

        qcs_vars.append(
            qcs.Array(
                name,
                dtype=float,
                value=values,
            )
        )

    # classification_refs
    classification_refs = []

    for label in qubit_labels:
        refs = variables_from_json[f"q_{label}"]["classification_refs"]

        real = np.asarray(refs["real"], dtype=float)
        imag = np.asarray(refs["imag"], dtype=float)

        classification_refs.append(real + 1j * imag)

    qcs_vars.append(
        qcs.Array(
            "classification_refs",
            dtype=complex,
            value=np.asarray(classification_refs, dtype=complex),
        )
    )

    return qcs_vars

def flattop_env(ramp_frac=0.1, n_edge=32):
    """
    Gaussian-edge flattop envelope (times normalized to 0..1.)
    """
    s = np.linspace(-2.0, 0.0, n_edge)
    edge = np.exp(-s**2 / 2.0)
    times = np.concatenate([np.linspace(0.0, ramp_frac, n_edge),
                           np.linspace(1.0 - ramp_frac, 1.0, n_edge)])
    amps = np.concatenate([edge, edge[::-1]])
    return qcs.ArbitraryEnvelope(times = times, amplitudes = amps)

class MyCalibrationSet(CalibrationSet):
    def __init__(self, topology, channels, variables, edge_list = None):
        # Initialize the parent CalibrationSet with the given topology.
        super().__init__(topology)

        # Use the first qudit group defined in the topology as the active qubits.
        # For example, this may correspond to labels such as (0, 1, 2).
        self.qubits = topology.qudits[0]
        self.edge_list = list(edge_list) if edge_list is not None else []

        # Register QCS variables in the CalibrationSet.
        #
        # init_vars is expected to be a list of qcs.Array objects, e.g.
        # [
        #     qcs.Array("x90_freq", ...),
        #     qcs.Array("x90_amp", ...),
        #     ...
        #     qcs.Array("classification_refs", ...),
        # ]
        #
        # After calling self.add_variable(var), the variable is automatically
        # attached to self.variables using its name.
        #
        # For example:
        #     self.add_variable(qcs.Array("x90_freq", ...))
        #
        # enables:
        #     self.variables.x90_freq
        #
        # Therefore, the pulse parameters can later be accessed as
        # self.variables.x90_freq, self.variables.ro_amp, etc.
        for var in variables:
            self.add_variable(var)

        # Convert the channel list into a dictionary for name-based access.
        # Avoid using self.channels because CalibrationSet may already define it.
        self.channel_map = {ch.name: ch for ch in channels}

        # Assign commonly used hardware channels.
        self.xy_awg = self._get_channel("xy_pulse")
        self.readout_awg = self._get_channel("readout_channels")
        self.dig = self._get_channel("acquire_channels")

        # Automatically register the default single-qubit sx gate
        # and measurement operation when the calibration set is created.
        self.add_sx()
        self.add_rz()
        if self.edge_list:
            self.add_cr()
        self.add_sx_ef()
        self.add_measurement()

    def _get_channel(self, name):
        """
        Return a channel by its name.

        Raises
        ------
        ValueError
            If the requested channel name does not exist in channel_map.
        """
        if name not in self.channel_map:
            available = list(self.channel_map.keys())
            raise ValueError(
                f"Channel '{name}' not found. Available channels: {available}"
            )

        return self.channel_map[name]

    def add_sx(self):
        """
        Add the sx gate using the x90 pulse parameters registered by add_variable.

        Required variables include:
        sx_ramp, sx_dur, sx_amp, sx_freq
        """
      
        # Build a flattop RF waveform for the x90 pulse.
        # Total duration = rise + hold + fall.
        x90_pulse = RFWaveform.create_rf_flattop(
            rise_duration=self.variables.sx_ramp,
            hold_duration=self.variables.sx_dur - 2 * self.variables.sx_ramp,
            fall_duration=self.variables.sx_ramp,
            envelope=GaussianEnvelope(),
            amplitude=self.variables.sx_amp,
            frequency=self.variables.sx_freq,
        )

        # Register the sx gate in the calibration set.
        GATES.x90.name='SX'
        self.add_sq_gate(
            "sx",
            GATES.x90,
            x90_pulse,
            self.qubits,
            self.xy_awg,
        )

        return self
    
    def add_sy(self):
        """
        Add the sy gate using the x90 pulse parameters registered by add_variable and adding 90 degree phase.

        Required variables include:
        sx_ramp, sx_dur, sx_amp, sx_freq
        """
      
        # Build a flattop RF waveform for the x90 pulse.
        # Total duration = rise + hold + fall.
        y90_pulse = RFWaveform.create_rf_flattop(
            rise_duration=self.variables.sx_ramp,
            hold_duration=self.variables.sx_dur - 2 * self.variables.sx_ramp,
            fall_duration=self.variables.sx_ramp,
            envelope=GaussianEnvelope(),
            amplitude=self.variables.sx_amp,
            frequency=self.variables.sx_freq,
            instantaneous_phase=np.pi/2
        )

        # Register the sx gate in the calibration set.
        GATES.y90.name='SY'
        self.add_sq_gate(
            "sy",
            GATES.y90,
            y90_pulse,
            self.qubits,
            self.xy_awg,
        )

        return self
    
    def add_rz(self):
        """
        Add the virtual z gate using the phi parameter registered by add_variable.

        Required variables include:
        phi
        """
        
        ####################################################################################
        #
        # Add the virtual Z gate (RZ)
        #
        ####################################################################################

        self.add_sq_gate(
            "rz",
            ParameterizedGate(PAULIS.rz, self.variables.phi),
            PhaseIncrement(self.variables.phi),
            self.qubits,
            self.xy_awg,
        )

    def add_cr(self, edge_list=None, name="cr"):
        """
        Add the cross-resonance gate on the given directed edges.

        Required variables include:
        cr_dur, cr_amp.
        
        Need to be called after add_sx().
        """

        if "sx" not in self.linkers:
            raise RuntimeError("add_cr requires the 'sx' linker: call add_sx() first.")
        
        edges = edge_list if edge_list is not None else self.edge_list
        if not edges:
            raise RuntimeError("No edges given: pass edge_list as an argument or set self.edge_list.")
        pairs = self.qubits.make_connectivity(edges)
        n_edges = len(edges)

        zx = PAULIS.sigma_z & PAULIS.sigma_x
        cr_gate = ParametricGate([zx], ["beta"])
        angles = Array(name = "beta", shape = (n_edges,), dtype = float)
        cr_param_gate = ParameterizedGate(cr_gate, angles)

        target_idx = [edge[1] for edge in edges]
        cr_freq = Array("cr_target_freq", dtype=float,
                        value=[self.variables.sx_freq.value[i] for i in target_idx])
        # in case of changing sx_freq, cr_freq should be changed manually, too.

        control_pulse = RFWaveform(
            self.variables.cr_dur,
            flattop_env(ramp_frac=0.1),
            self.variables.cr_amp,
            cr_freq,
        )
        self.add_cr_gate(
            cr_param_gate,
            pairs,
            sq_linker="sx",
            control_waveform=control_pulse,   
            name = name, 
        )
        return self    

    def add_sizzle(self, edge_list=None, name="sizzle"):
        """
        Add the siZZle gate on the given directed edges.
        Required variables include:
        sizzle_dur, sizzle_freq, sizzle_c_amp, sizzle_t_amp, sizzle_phase.
        sizzle_phase is the phase of the target drive relative to the control drive.
        
        Need to be called after add_sx().

        example usage: 
            prog.add_parametric_gate(
                cal.sizzle_gate.gate, [cal.sizzle_gate.parameters[0]], pair
                )
        
        """

        if "sx" not in self.linkers:
            raise RuntimeError("add_sizzle requires the 'sx' linker: call add_sx() first.")

        edges = edge_list if edge_list is not None else self.edge_list
        if not edges:
            raise RuntimeError("No edges given: pass edge_list as an argument or set self.edge_list.")
        pairs = self.qubits.make_connectivity(edges)
        n_edges = len(edges)

        zz = PAULIS.sigma_z & PAULIS.sigma_z
        sizzle_gate = ParametricGate([zz], ["beta"])
        angles = Array(name="sizzle_beta", shape=(n_edges,), dtype=float)
        sizzle_param_gate = ParameterizedGate(sizzle_gate, angles)

        control_idx = [edge[0] for edge in edges]
        target_idx = [edge[1] for edge in edges]

        control_pulse = RFWaveform(
            self.variables.sizzle_dur,
            flattop_env(ramp_frac=0.1),
            self.variables.sizzle_c_amp,
            self.variables.sizzle_freq,
        )

        target_pulse = RFWaveform(
            self.variables.sizzle_dur,
            flattop_env(ramp_frac=0.1),
            self.variables.sizzle_t_amp,
            self.variables.sizzle_freq,
            instantaneous_phase=self.variables.sizzle_phase,
        )

        sizzle_prog = qcs.Program()
        sizzle_prog.add_waveform(control_pulse, self.xy_awg[control_idx])
        sizzle_prog.add_waveform(target_pulse, self.xy_awg[target_idx])

        sizzle_linker = qcs.ParameterizedLinker(sizzle_param_gate, pairs, sizzle_prog)

        self.add_linker(name, sizzle_linker)
        return self

    def add_sx_ef(self):
        """
        Add the sx_ef gate using the x90 pulse parameters registered by add_variable.

        Required variables include:
        sx_ef_ramp, sx_ef_dur, sx_ef_amp, sx_ef_freq
        """
      
        # Build a flattop RF waveform for the x90 pulse.
        # Total duration = rise + hold + fall.
        x90_ef_pulse = RFWaveform.create_rf_flattop(
            rise_duration=self.variables.sx_ef_ramp,
            hold_duration=self.variables.sx_ef_dur - 2 * self.variables.sx_ef_ramp,
            fall_duration=self.variables.sx_ef_ramp,
            envelope=GaussianEnvelope(),
            amplitude=self.variables.sx_ef_amp,
            frequency=self.variables.sx_ef_freq,
        )

        # Register the sx gate in the calibration set.
        self.add_sq_gate(
            "sx_ef",
            qcs.Gate([[1,0],[0,1]],name='SXef'),
            x90_ef_pulse,
            self.qubits,
            self.xy_awg,
        )

        return self

    def add_measurement(self):
        """
        Add the measurement linker using variables registered by add_variable.

        This creates a replacement program containing:
        1. readout waveform on the readout channel
        2. acquisition filter on the digitizer/acquire channel
        3. classifier references for state discrimination

        Required variables include:
        ro_freq, ro_amp, ro_dur, ro_delay, ro_phase, acq_dur, acq_delay, classification_refs
        """

        # Readout drive pulse.
        readout_pulse = RFWaveform(
            self.variables.ro_dur,
            ConstantEnvelope(),
            self.variables.ro_amp,
            self.variables.ro_freq,
            self.variables.ro_phase,
        )

        # Integration filter used for acquisition.
        # The amplitude is set to 1 because this waveform acts as a filter.
        integration_filter = RFWaveform(
            self.variables.acq_dur,
            ConstantEnvelope(),
            1,
            self.variables.ro_freq,
        )

        # Program that replaces the abstract Measure operation.
        replacement_program = qcs.Program()

        # Add the readout pulse to the readout AWG channel.
        replacement_program.add_waveform(
            readout_pulse,
            self.readout_awg,
            new_layer=True,
            pre_delay=self.variables.ro_delay,
        )

        # Build the classifier from qubit-dependent reference points.
        # classification_refs is also registered through self.add_variable(...),
        # so it is accessed as self.variables.classification_refs.
        classifiers = qcs.Classifier(self.variables.classification_refs)

        # Add acquisition with integration filter and classifier.
        # The acquisition starts after readout delay plus acquisition delay.
        replacement_program.add_acquisition(
            integration_filter,
            self.dig,
            classifiers,
            pre_delay=self.variables.ro_delay + self.variables.acq_delay,
        )

        # Abstract measurement operation to be linked to the replacement program.
        measure = qcs.Measure()

        # Link the abstract Measure operation to the hardware-level program.
        meas_linker = qcs.ParameterizedLinker(
            measure,
            self.qubits,
            replacement_program,
        )

        # Register the measurement linker in the calibration set.
        self.add_linker("measurement", meas_linker)

        return self