import os
import joblib
import tempfile

import mlflow as mlf
from hydra.utils import get_original_cwd
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

def set_mlflow(
    exp_name, exp_tags=None, run_tags=None, run_id=None, get_last_run=False
):
    mlf.set_tracking_uri(f"file://{get_original_cwd()}/mlruns")
    client = mlf.tracking.MlflowClient()
    experiment = mlf.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = client.create_experiment(name=exp_name)
        experiment = mlf.get_experiment(experiment_id)
        if exp_tags is not None:
            for name, tag in exp_tags.items():
                mlf.set_experiment_tag(experiment_id, name, tag)
    else:
        experiment_id = experiment.experiment_id
    if run_id is None and not get_last_run:
        run_tags = {} if run_tags is None else run_tags
        run_tags = mlf.tracking.context.registry.resolve_tags(run_tags)
        run = client.create_run(experiment_id=experiment_id, tags=run_tags)
    elif not get_last_run:
        run = mlf.get_run(run_id=run_id)
    else:
        run = mlf.search_runs(experiment_id, output_format="list")[0]
    return run

