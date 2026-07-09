us=1e-6
ns=1e-9
MHz=1e6
GHz=1e9
kHz=1e3

from sklearn.mixture import GaussianMixture
import keysight.qcs as qcs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq
from scipy.stats import norm



def _rotate_data(data: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate 2D IQ data by a given angle in radians.
    """
    rot_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)],
        ]
    )
    return data @ rot_matrix.T


def _get_angle(vec: np.ndarray) -> float:
    """
    Return the angle of a 2D vector from the origin.
    """
    return np.arctan2(vec[1], vec[0])

def IQ_fit(
    iq_array,
    sweep_idx,
    qubits,
):
    """
    Fit IQ measurement data for one or more qubits.

    Parameters
    ----------
    program:
        Object that has:
            - program.get_iq_array(...)
    qubits:
        Qubits to analyze.

    Returns
    -------
    fit_results:
        dict mapping qubit index to fit result.
    shots_dict:
        dict containing raw and rotated shots.
    """

    array = iq_array

    fit_results = {}
    shots_dict = {}

    for qubit in qubits:
        # qubit index 처리
        if hasattr(qubit, "labels"):
            q_idx = qubit.labels[0]
        else:
            q_idx = int(qubit)

        array_qb = array.sel(qudit=f"{qubit}")

        # ground / excited IQ samples
        shots_g_complex = array_qb.isel({sweep_idx: 0}).data
        shots_e_complex = array_qb.isel({sweep_idx: 1}).data

        shots_g = np.array([[z.real, z.imag] for z in shots_g_complex])
        shots_e = np.array([[z.real, z.imag] for z in shots_e_complex])

        # 평균 IQ 위치
        mu_g = np.mean(shots_g, axis=0)
        mu_e = np.mean(shots_e, axis=0)

        # ground -> excited 방향을 I축에 맞추기 위한 회전각
        rotation_angle = _get_angle(mu_e - mu_g)

        # IQ cloud 회전
        shots_g_rot = _rotate_data(shots_g, -rotation_angle)
        shots_e_rot = _rotate_data(shots_e, -rotation_angle)

        # I축 projection
        shots_g_1d = shots_g_rot[:, 0]
        shots_e_1d = shots_e_rot[:, 0]

        # Gaussian fit
        mu_g_1d, sigma_g = norm.fit(shots_g_1d)
        mu_e_1d, sigma_e = norm.fit(shots_e_1d)

        def diff_pdf(x):
            return (
                norm.pdf(x, mu_g_1d, sigma_g)
                - norm.pdf(x, mu_e_1d, sigma_e)
            )

        # decision boundary 계산
        try:
            x_min = min(mu_g_1d, mu_e_1d) - 3 * max(sigma_g, sigma_e)
            x_max = max(mu_g_1d, mu_e_1d) + 3 * max(sigma_g, sigma_e)
            decision_boundary = brentq(diff_pdf, x_min, x_max)
        except ValueError:
            decision_boundary = (mu_g_1d + mu_e_1d) / 2

        # fidelity 계산
        if mu_g_1d < mu_e_1d:
            p_e_given_g = 1 - norm.cdf(decision_boundary, mu_g_1d, sigma_g)
            p_g_given_e = norm.cdf(decision_boundary, mu_e_1d, sigma_e)
        else:
            p_e_given_g = norm.cdf(decision_boundary, mu_g_1d, sigma_g)
            p_g_given_e = 1 - norm.cdf(decision_boundary, mu_e_1d, sigma_e)

        fidelity = 1 - 0.5 * (p_e_given_g + p_g_given_e)

        fit_results[q_idx] = {
            "mu_g": mu_g_1d,
            "mu_e": mu_e_1d,
            "sigma_g": sigma_g,
            "sigma_e": sigma_e,
            "decision_boundary": decision_boundary,
            "fidelity": fidelity,
            "rotation_angle": rotation_angle,
        }

        shots_dict[q_idx] = {
            "shots_g": shots_g,
            "shots_e": shots_e,
            "shots_g_rot": shots_g_rot,
            "shots_e_rot": shots_e_rot,
        }

    return fit_results, shots_dict


def get_RO_fidelity(
    iq_array,
    sweep_idx,
    qubits,
):
    """
    Return readout fidelity for each qubit.
    """

    fit_results, _ = IQ_fit(
        iq_array=iq_array,
        sweep_idx=sweep_idx,
        qubits=qubits,
    )

    return {
        qubit: result["fidelity"]
        for qubit, result in fit_results.items()
    }

def IQ_plot(
    iq_array,
    sweep_idx,
    qubits,
    n_bins: int = 100,
):
    """
    Plot IQ discrimination data.
    """

    fit_results, shots_dict_all = IQ_fit(
        iq_array=iq_array,
        sweep_idx=sweep_idx,
        qubits=qubits,
    )

    qubit_indices = sorted(fit_results.keys())

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Original IQ Cloud",
            "Rotated IQ Cloud",
            "1D Histogram",
            "Gaussian Fit",
        ),
        horizontal_spacing=0.2,
        vertical_spacing=0.2,
    )

    buttons = []
    num_traces_per_qubit = 9

    for i, qubit in enumerate(qubit_indices):
        fit_dict = fit_results[qubit]
        shots_dict = shots_dict_all[qubit]

        shots_g = shots_dict["shots_g"]
        shots_e = shots_dict["shots_e"]
        shots_g_rot = shots_dict["shots_g_rot"]
        shots_e_rot = shots_dict["shots_e_rot"]

        mu_g_1d = fit_dict["mu_g"]
        mu_e_1d = fit_dict["mu_e"]
        sigma_g = fit_dict["sigma_g"]
        sigma_e = fit_dict["sigma_e"]
        decision_boundary = fit_dict["decision_boundary"]
        fidelity = fit_dict["fidelity"]

        shots_g_1d = shots_g_rot[:, 0]
        shots_e_1d = shots_e_rot[:, 0]

        x_vals = np.linspace(
            min(shots_g_1d.min(), shots_e_1d.min()),
            max(shots_g_1d.max(), shots_e_1d.max()),
            1000,
        )

        y_g = norm.pdf(x_vals, mu_g_1d, sigma_g)
        y_e = norm.pdf(x_vals, mu_e_1d, sigma_e)
        y_max = max(y_g.max(), y_e.max())

        visible = i == 0

        traces = [
            go.Scatter(
                x=shots_g[:, 0],
                y=shots_g[:, 1],
                mode="markers",
                marker={"color": "blue", "size": 4},
                name="|g⟩ original",
                visible=visible,
            ),
            go.Scatter(
                x=shots_e[:, 0],
                y=shots_e[:, 1],
                mode="markers",
                marker={"color": "red", "size": 4},
                name="|e⟩ original",
                visible=visible,
            ),
            go.Scatter(
                x=shots_g_rot[:, 0],
                y=shots_g_rot[:, 1],
                mode="markers",
                marker={"color": "blue", "size": 4},
                name="|g⟩ rotated",
                visible=visible,
                showlegend=False,
            ),
            go.Scatter(
                x=shots_e_rot[:, 0],
                y=shots_e_rot[:, 1],
                mode="markers",
                marker={"color": "red", "size": 4},
                name="|e⟩ rotated",
                visible=visible,
                showlegend=False,
            ),
            go.Histogram(
                x=shots_g_1d,
                nbinsx=n_bins,
                marker_color="blue",
                opacity=0.6,
                name="|g⟩ 1D",
                visible=visible,
            ),
            go.Histogram(
                x=shots_e_1d,
                nbinsx=n_bins,
                marker_color="red",
                opacity=0.6,
                name="|e⟩ 1D",
                visible=visible,
            ),
            go.Scatter(
                x=x_vals,
                y=y_g,
                mode="lines",
                name="Fit |g⟩",
                line={"color": "blue", "dash": "dash"},
                visible=visible,
            ),
            go.Scatter(
                x=x_vals,
                y=y_e,
                mode="lines",
                name="Fit |e⟩",
                line={"color": "red", "dash": "dash"},
                visible=visible,
            ),
            go.Scatter(
                x=[decision_boundary, decision_boundary],
                y=[0, y_max],
                mode="lines",
                name="Decision Boundary",
                line={"color": "black", "dash": "dot"},
                visible=visible,
            ),
        ]

        subplot_mapping = [
            (1, 1),
            (1, 1),
            (1, 2),
            (1, 2),
            (2, 1),
            (2, 1),
            (2, 2),
            (2, 2),
            (2, 2),
        ]

        for trace, (row, col) in zip(traces, subplot_mapping):
            fig.add_trace(trace, row=row, col=col)

        visibility = [False] * (len(qubit_indices) * num_traces_per_qubit)
        start_idx = i * num_traces_per_qubit

        for j in range(num_traces_per_qubit):
            visibility[start_idx + j] = True

        buttons.append(
            dict(
                label=f"Select qubit: {qubit}",
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": (
                            f"IQ Discrimination — Qubit {qubit} — "
                            f"Fidelity: {fidelity:.4f}"
                        )
                    },
                ],
            )
        )

    first_qubit = qubit_indices[0]

    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 1.2,
                "y": 1.10,
            }
        ],
        height=800,
        width=1000,
        title=(
            f"IQ Discrimination — Qubit {first_qubit} — "
            f"Fidelity: {fit_results[first_qubit]['fidelity']:.4f}"
        ),
        bargap=0.05,
    )

    fig.update_xaxes(title_text="I", row=1, col=1)
    fig.update_yaxes(title_text="Q", row=1, col=1)

    fig.update_xaxes(title_text="I rotated", row=1, col=2)
    fig.update_yaxes(title_text="Q rotated", row=1, col=2)

    fig.update_xaxes(title_text="Projected I quadrature", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    fig.update_xaxes(title_text="Projected I quadrature", row=2, col=2)
    fig.update_yaxes(title_text="PDF", row=2, col=2)

    return fig




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
    
def set_value_at(var, value, index):
    v = var.value.copy()

    v[index] = value

    var.value = v

