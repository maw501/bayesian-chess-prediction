import pickle

import numpy as np


def get_class_probs(params, param_name, class_levels=[1, 2, 3]):
    out = []
    for level in class_levels:
        out.append(np.sum(params[param_name] == level, 0))
    out = np.array(out).T
    out_prob = out / np.mean(out.sum(1))
    return out_prob


def print_results(df, cls_pred, prefix, dp=2):
    d0 = np.mean(np.abs(df["p1_outcome"] - cls_pred) == 0)
    d1 = np.mean(np.abs(df["p1_outcome"] - cls_pred) == 1)
    d2 = np.mean(np.abs(df["p1_outcome"] - cls_pred) == 2)
    print(
        f"{prefix} - exact: {d0:.{dp}f}, out by one: {d1:.{dp}f}, out by two: {d2:.{dp}f}"  # noqa
    )


def save_model(file_name, model):
    with open(str(file_name), "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    return pickle.load(open(str(file_name), "rb"))
