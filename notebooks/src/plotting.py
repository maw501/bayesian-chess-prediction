import matplotlib.pyplot as plt
import numpy as np


def plot_expected_outcomes(ypred, ranks, player_1_white):
    rank_diffs = ranks[:, 1] - ranks[:, 0]
    p1_mask = player_1_white == 1

    out = np.array([np.sum(ypred == l, 0) for l in [1, 2, 3]]).T
    out_prob = out / np.mean(out.sum(1))
    expected_outcomes = [np.round(i, 2) for i in out_prob.mean(0)]
    expected_outcomes_white = [
        np.round(i, 2) for i in out_prob[p1_mask].mean(0)
    ]
    expected_outcomes_black = [
        np.round(i, 2) for i in out_prob[~p1_mask].mean(0)
    ]

    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=False)
    ax = ax.ravel()

    ax[0].scatter(
        rank_diffs[p1_mask],
        out_prob[:, 2][p1_mask],
        label="win",
        color="lightsteelblue",
    )
    ax[1].scatter(
        rank_diffs[~p1_mask],
        out_prob[:, 2][~p1_mask],
        label="win",
        color="lightsteelblue",
    )

    ax[0].scatter(
        rank_diffs[p1_mask],
        out_prob[:, 0][p1_mask],
        label="lose",
        color="lightcoral",
    )
    ax[1].scatter(
        rank_diffs[~p1_mask],
        out_prob[:, 0][~p1_mask],
        label="lose",
        color="lightcoral",
    )

    ax[0].scatter(
        rank_diffs[p1_mask],
        out_prob[:, 1][p1_mask],
        label="draw",
        color="lightgreen",
    )
    ax[1].scatter(
        rank_diffs[~p1_mask],
        out_prob[:, 1][~p1_mask],
        label="draw",
        color="lightgreen",
    )

    ax[0].set_title(f"P1 as white [L, D, W]: {expected_outcomes_white}")
    ax[1].set_title(f"P1 as black [L, D, W]: {expected_outcomes_black}")

    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)

    ax[0].legend(loc="upper left")
    ax[1].legend(loc="upper left")
    fig.text(
        0.5,
        0.04,
        "P1 rank advantage [left is  P2 stronger, right P1 stronger]",
        ha="center",
    )
    fig.text(
        0.075, 0.5, "Probability of outcome", va="center", rotation="vertical"
    )

    plt.suptitle(f"Expected outcomes for P1 [L, D, W]: {expected_outcomes}")


def fit_plot_prior(
    override_params, base_param_dict, model, ranks_fake, player_1_white
):
    stan_data = base_param_dict.copy()
    stan_data.update(override_params)
    fit = model.sampling(data=stan_data)
    params = fit.extract(permuted=True)
    plot_expected_outcomes(params["ypred"], ranks_fake, player_1_white)
    return params
