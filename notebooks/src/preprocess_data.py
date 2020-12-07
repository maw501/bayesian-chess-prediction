import numpy as np
import pandas as pd

pd.set_option("mode.chained_assignment", None)


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
        raise Exception("What the fluff.")


def load_df_drop_nas(file_path):
    df = pd.read_csv(file_path, parse_dates=["date"])
    return df.dropna()


def decide_when_p1_is_white(df):
    df = df.copy()
    np.random.seed(42)
    p1_white = np.random.binomial(1, 0.5, size=len(df)) == 1
    df["p1_white"] = p1_white
    return df


def get_scores_from_string(df):
    df = df.copy()
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
    return df


def create_single_p1_p2_col(df, suffix):
    df = df.copy()
    p1_white = df["p1_white"].values
    df.loc[p1_white, f"p1_{suffix}"] = df.loc[p1_white, f"white_{suffix}"]
    df.loc[~p1_white, f"p1_{suffix}"] = df.loc[~p1_white, f"black_{suffix}"]

    df.loc[p1_white, f"p2_{suffix}"] = df.loc[p1_white, f"black_{suffix}"]
    df.loc[~p1_white, f"p2_{suffix}"] = df.loc[~p1_white, f"white_{suffix}"]
    return df


def create_p1_p2_cols(df):
    df = df.copy()
    suffixes = ["id", "elo", "title", "score"]
    for suffix in suffixes:
        df = create_single_p1_p2_col(df, suffix)
    df["p1_id"] = df["p1_id"].astype(np.uint64)
    df["p2_id"] = df["p2_id"].astype(np.uint64)
    return df


def get_id_to_elo_rating_dict(df):
    elo_df = pd.concat(
        [
            df[["p1_id", "p1_elo"]].rename(
                columns={"p1_id": "id", "p1_elo": "elo"}
            ),
            df[["p2_id", "p2_elo"]].rename(
                columns={"p2_id": "id", "p2_elo": "elo"}
            ),
        ]
    )

    id_to_elo_rating_df = elo_df.groupby("id").mean().reset_index()
    id_to_elo_rating_df = id_to_elo_rating_df.sort_values(
        "elo", ascending=False
    )
    id_to_elo_rating_df["rank"] = range(1, len(id_to_elo_rating_df) + 1)

    return (
        id_to_elo_rating_df[["rank", "id"]].set_index("id").to_dict()["rank"]
    )


def add_rank_cols(df, id_to_elo_rating_dict):
    df = df.copy()
    df["p1_prior_rank"] = df["p1_id"].apply(lambda x: id_to_elo_rating_dict[x])
    df["p2_prior_rank"] = df["p2_id"].apply(lambda x: id_to_elo_rating_dict[x])

    df["abs_rank_diff"] = np.abs(
        df["p1_prior_rank"].values - df["p2_prior_rank"].values
    )
    return df


def add_p1_outcome_col(df):
    df = df.copy()
    df["p1_outcome"] = (1 + df["p1_score"] * 2).astype(int)
    return df


def create_df(file_path):
    df = load_df_drop_nas(file_path)
    df = (
        df.pipe(decide_when_p1_is_white)
        .pipe(get_scores_from_string)
        .pipe(create_p1_p2_cols)
    )
    return df


def process_train_df(train):
    train = train.copy()
    id_to_elo_rating_dict = get_id_to_elo_rating_dict(train)
    train = add_rank_cols(train, id_to_elo_rating_dict)
    train = add_p1_outcome_col(train)
    return train
