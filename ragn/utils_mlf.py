import os
import yaml
import tempfile
from os.path import join, abspath
from os import getcwd, walk

import joblib
import pandas as pd
import mlflow as mlf
from prettytable import PrettyTable
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig


__all__ = ["set_mlflow", "log_params_from_omegaconf_dict"]


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                mlf.log_param(f"{parent_name}.{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlf.log_param(f"{parent_name}.{i}", v)


def save_pickle(name, obj):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            joblib.dump(obj, f)
        mlf.log_artifact(path)


def load_pickle(name):
    path = mlf.get_artifact_uri(f"{name}.pkl").split(":")[-1]
    return joblib.load(path)


def set_mlflow(exp_name, fix_path=False, load_runs=False):
    _ = mlf_fix_artifact_path() if fix_path else None
    mlf.set_tracking_uri(f"file://{to_absolute_path('mlruns')}")
    client = mlf.tracking.MlflowClient()
    experiment = mlf.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = client.create_experiment(name=exp_name)
        experiment = mlf.get_experiment(experiment_id)
    else:
        experiment_id = experiment.experiment_id
    if not load_runs:
        run = client.create_run(experiment_id=experiment_id)
    else:
        runs_df = mlf_get_run_list(experiment_id)
        if runs_df.empty:
            return None
        t = PrettyTable([""] + runs_df.columns.tolist())
        for i, row in runs_df.iterrows():
            table_row = [
                i,
                row["Status"],
                row["Start Time"],
                "{:.3f}".format(row["Duration (h)"]),
                "{:.3f}".format(row["Final Loss"]),
            ]
            for i in range(4, runs_df.shape[1]):
                table_row.append("{:.3f}".format(row.iloc[i]))
            t.add_row(table_row)
        print(t)
        idx = int(input("Choose the number of the desired run: "))
        run_id = runs_df.loc[idx, "Run ID"]
        run = mlf.get_run(run_id=run_id)
    return run


def mlf_fix_artifact_path():
    def update_artifact_path(meta, key):
        if key in meta:
            artifact_path = meta[key]
            new_artifact_path = join(
                "file://" + cwd, artifact_path[artifact_path.find("mlruns") :]
            )
            meta[key] = new_artifact_path

    cwd = abspath(getcwd())
    for root, _, files in walk(join(cwd, "mlruns")):
        for f_name in files:
            if f_name == "meta.yaml":
                path = join(cwd, "mlruns", root, f_name)
                with open(path, "r") as f:
                    meta = yaml.safe_load(f)
                update_artifact_path(meta, "artifact_location")
                update_artifact_path(meta, "artifact_uri")
                with open(path, "w") as f:
                    meta = yaml.dump(meta, f)


def mlf_get_run_list(experiment_id):
    def get_duration(row):
        if row["start_time"] is None or row["end_time"] is None:
            return None
        return (row["end_time"] - row["start_time"]).total_seconds() / 3600

    data = {}
    metric_columns = []
    runs_df = mlf.search_runs(experiment_ids=[experiment_id], output_format="pandas")
    for c in runs_df.columns:  # type: ignore
        if c.startswith("params.training.metrics"):
            metric_columns.append(c)
    metrics = (
        runs_df[metric_columns]  # type: ignore
        .melt()
        .loc[lambda d: ~d["value"].isnull()]
        .drop_duplicates(subset=["value"])
        .set_index("variable")["value"]
    )
    data["Run ID"] = runs_df["run_id"].tolist()  # type: ignore
    data["Status"] = runs_df["status"].tolist()  # type: ignore
    data["Start Time"] = (
        runs_df["start_time"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S")).tolist()  # type: ignore
    )
    data["Duration (h)"] = (
        runs_df[["start_time", "end_time"]].apply(get_duration, axis=1).tolist()  # type: ignore
    )
    data["Final Loss"] = runs_df["metrics.loss"].tolist()  # type: ignore
    for m in metrics:  # FIXME: !!
        try:
            data[f"Final {m} MMD"] = runs_df[f"metrics.mmd_{m}"].tolist()  # type: ignore
        except KeyError:
            pass
    runs_df = pd.DataFrame(data)
    return (
        runs_df.loc[~runs_df.iloc[:, 5:].isna().all(axis=1)]
        .sort_values("Start Time", ascending=False)
        .reset_index(drop=True)
    )
