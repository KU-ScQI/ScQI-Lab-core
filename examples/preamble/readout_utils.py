import numpy as np
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture

def fit_gmm_with_centers(
    iq_data,
    centers_init,
    covariance_type="full",
    reg_covar=1e-12,
    max_iter=1000,
    random_state=0,
):
    """
    User-defined initial centers를 이용해서 IQ blob들을 GMM으로 분리한다.

    Parameters
    ----------
    iq_data:
        IQ 데이터. shape = (n_shots, 2)
        iq_data[:, 0] = I
        iq_data[:, 1] = Q

    centers_init:
        초기 blob center. shape = (n_components, 2)
        예:
        np.array([
            [-0.00055,  0.00025],
            [ 0.00010,  0.00035],
            [ 0.00050, -0.00010],
        ])

    covariance_type:
        "full", "diag", "tied", "spherical" 중 선택.
        보통 "full" 추천.

    reg_covar:
        covariance regularization.
        IQ 값이 1e-4 ~ 1e-3 스케일이면 1e-12 정도부터 시도.

    Returns
    -------
    result:
        dict containing gmm, labels, probabilities, means, covariances, weights.
    """

    iq_data = np.asarray(iq_data)
    centers_init = np.asarray(centers_init)

    if iq_data.ndim != 2 or iq_data.shape[1] != 2:
        raise ValueError("iq_data must have shape (n_shots, 2).")

    if centers_init.ndim != 2 or centers_init.shape[1] != 2:
        raise ValueError("centers_init must have shape (n_components, 2).")

    n_components = len(centers_init)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        means_init=centers_init,
        n_init=1,
        max_iter=max_iter,
        reg_covar=reg_covar,
        random_state=random_state,
    )

    gmm.fit(iq_data)

    labels = gmm.predict(iq_data)
    probs = gmm.predict_proba(iq_data)

    result = {
        "gmm": gmm,
        "iq_data": iq_data,
        "labels": labels,
        "probs": probs,
        "means": gmm.means_,
        "covariances": gmm.covariances_,
        "weights": gmm.weights_,
        "centers_init": centers_init,
    }

    return result

def plot_gmm_result(
    result,
    show_initial_centers=True,
    marker_size=4,
):
    """
    GMM clustering 결과를 IQ plane에 표시한다.
    """

    iq_data = result["iq_data"]
    labels = result["labels"]
    means = result["means"]
    centers_init = result["centers_init"]

    n_components = len(means)

    fig = go.Figure()

    for k in range(n_components):
        mask = labels == k

        fig.add_trace(
            go.Scatter(
                x=iq_data[mask, 0],
                y=iq_data[mask, 1],
                mode="markers",
                marker=dict(size=marker_size),
                name=f"Blob {k}",
            )
        )

    if show_initial_centers:
        fig.add_trace(
            go.Scatter(
                x=centers_init[:, 0],
                y=centers_init[:, 1],
                mode="markers",
                marker=dict(
                    size=14,
                    symbol="cross",
                    color="gray",
                ),
                name="Initial centers",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=means[:, 0],
            y=means[:, 1],
            mode="markers",
            marker=dict(
                size=18,
                symbol="x",
                color="black",
            ),
            name="Fitted centers",
        )
    )

    fig.update_layout(
        width=750,
        height=650,
        title="GMM clustering of IQ blobs",
        xaxis_title="I",
        yaxis_title="Q",
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig

def covariance_ellipse(mean, cov, n_std=2.0, n_points=200):
    """
    2D Gaussian covariance ellipse 생성.
    """

    eigvals, eigvecs = np.linalg.eigh(cov)

    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.array([np.cos(theta), np.sin(theta)])

    ellipse = eigvecs @ np.diag(np.sqrt(eigvals) * n_std) @ circle
    ellipse = ellipse.T + mean

    return ellipse


def plot_gmm_result_with_ellipses(
    result,
    n_std=2.0,
    show_initial_centers=True,
    marker_size=4,
):
    """
    GMM clustering 결과와 Gaussian ellipse를 함께 표시한다.
    """

    iq_data = result["iq_data"]
    labels = result["labels"]
    means = result["means"]
    covariances = result["covariances"]
    centers_init = result["centers_init"]

    n_components = len(means)

    fig = go.Figure()

    for k in range(n_components):
        mask = labels == k

        fig.add_trace(
            go.Scatter(
                x=iq_data[mask, 0],
                y=iq_data[mask, 1],
                mode="markers",
                marker=dict(size=marker_size),
                name=f"Blob {k}",
            )
        )

        ellipse = covariance_ellipse(
            mean=means[k],
            cov=covariances[k],
            n_std=n_std,
        )

        fig.add_trace(
            go.Scatter(
                x=ellipse[:, 0],
                y=ellipse[:, 1],
                mode="lines",
                name=f"Blob {k} ellipse",
                showlegend=True,
            )
        )

    if show_initial_centers:
        fig.add_trace(
            go.Scatter(
                x=centers_init[:, 0],
                y=centers_init[:, 1],
                mode="markers",
                marker=dict(
                    size=14,
                    symbol="cross",
                    color="gray",
                ),
                name="Initial centers",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=means[:, 0],
            y=means[:, 1],
            mode="markers",
            marker=dict(
                size=18,
                symbol="x",
                color="black",
            ),
            name="Fitted centers",
        )
    )

    fig.update_layout(
        width=750,
        height=650,
        title=f"GMM clustering of IQ blobs with {n_std}σ ellipses",
        xaxis_title="I",
        yaxis_title="Q",
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig

def complex_to_array(num):
    return np.array([np.real(num),np.imag(num)]).transpose()

def array_to_complex(array):
    return np.array(array[0]+1j*array[1]).transpose()