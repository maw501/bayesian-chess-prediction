import pandas as pd
import numpy as np

DP = "../data/games.csv"

pd.set_option("mode.chained_assignment", None)


def load_data():
    return pd.read_csv(DP, parse_dates=["date"])


def get_result_integer(string_result, position):
    return float(string_result.split("-")[position][0])


def get_result_non_integer(string_result, position):
    res = [x.split("/") if "/" in x else x for x in string_result.split("-")]
    res = [i for s in res for i in s]
    if position == 0:
        return float(res[0]) / float(res[1])
    elif position == 1:
        return float(res[2]) / float(res[3])
    else:
        raise Exception("Argh")


def create_cols(df, mask, white_col, black_col, fave_name, underdog_name):
    df[fave_name] = df.loc[mask, white_col]
    df.loc[~mask, fave_name] = df.loc[~mask, black_col]
    df[underdog_name] = df.loc[~mask, white_col]
    df.loc[mask, underdog_name] = df.loc[mask, black_col]


def process_all_data():
    df = load_data()
    df = df.dropna()
    df["year"] = df["date"].dt.year

    df["white_score"] = df["result"].apply(
        lambda x: get_result_integer(x, 0)
        if "/" not in x
        else get_result_non_integer(x, 0)
    )
    df["black_score"] = df["result"].apply(
        lambda x: get_result_integer(x, 1)
        if "/" not in x
        else get_result_non_integer(x, 1)
    )
    white_win = df["white_score"] > df["black_score"]
    black_win = df["white_score"] < df["black_score"]
    draw = df["white_score"] == df["black_score"]
    df["outcome"] = 0
    df.loc[white_win, "outcome"] = "white"
    df.loc[black_win, "outcome"] = "black"
    df.loc[draw, "outcome"] = "draw"
    df["white_black_elo_diff"] = df["white_elo"] - df["black_elo"]
    return df


def create_train_data():
    df = process_all_data()
    train = df.loc[df["date"] < "2019-01-01"]

    train_ids = list(
        set(train["white_id"].values).union(set(train["black_id"].values))
    )

    train_id_to_avg_elo = (
        train.groupby("white_id")["white_elo"].mean().to_dict()
    )

    train_id_to_avg_elo_ranked = {
        k: v
        for k, v in sorted(
            train_id_to_avg_elo.items(), key=lambda item: item[1], reverse=True
        )
    }

    ranking_dict = {
        k: v
        for k, v in zip(
            train_id_to_avg_elo_ranked.keys(), range(1, len(train_ids) + 1)
        )
    }

    train["white_prior_rank"] = train["white_id"].apply(
        lambda x: ranking_dict[x]
    )
    train["black_prior_rank"] = train["black_id"].apply(
        lambda x: ranking_dict[x]
    )
    white_fave = train["white_prior_rank"] < train["black_prior_rank"]

    create_cols(
        train, white_fave, "white_id", "black_id", "fave_id", "underdog_id"
    )
    create_cols(
        train,
        white_fave,
        "white_prior_rank",
        "black_prior_rank",
        "fave_rank",
        "underdog_rank",
    )
    create_cols(
        train,
        white_fave,
        "white_score",
        "black_score",
        "fave_score",
        "underdog_score",
    )

    train["fave_outcome"] = (1 + train["fave_score"] * 2).astype(int)

    train["white_is_fave"] = (train["white_id"] == train["fave_id"]).astype(
        int
    )

    train["abs_rank_diff"] = np.abs(
        train["white_prior_rank"].values - train["black_prior_rank"].values
    )
    train["fave_underdog_score_diff"] = (
        train["fave_score"].values - train["underdog_score"].values
    )
    return train
