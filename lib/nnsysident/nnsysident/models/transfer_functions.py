from collections import OrderedDict
import torch
import pandas as pd
import warnings


def core_transfer(model, trained_model_table, trainer_config, seed, t_model_hash, t_dataset_hash, t_trainer_hash):
    assert (
        "detach_core" in trainer_config and trainer_config["detach_core"] is True
    ), "When transferring the core, 'detach_core' in the 'trainer_config' must be set to True."
    warnings.warn('This transfer function does not use the seed')
    # get all trained models corresponding to the combination of model, dataset and trainer
    restricted_trained_model_table = (
        trained_model_table
        & "model_hash = '{}'".format(t_model_hash)
        & "dataset_hash = '{}'".format(t_dataset_hash)
        & "trainer_hash = '{}'".format(t_trainer_hash)
    )
    trained_model_entries = pd.DataFrame(restricted_trained_model_table.fetch())
    # from this selection, filter out the trained model with the best score (filtering over seeds)
    trained_model_entry = trained_model_entries.loc[
        trained_model_entries["score"] == trained_model_entries["score"].max()
    ]
    # get the state dict of the best model
    state_dict = (
        restricted_trained_model_table * restricted_trained_model_table.ModelStorage
        & "seed = {}".format(int(trained_model_entry["seed"]))
    ).fetch1("model_state", download_path="models/")
    # extract the core parameters of the state dict and initialize the new model with it
    core_dict = OrderedDict([(k, v) for k, v in torch.load(state_dict).items() if k[0:5] == "core."])
    model.load_state_dict(core_dict, strict=False)


def core_transfer_by_seed(model, trained_model_table, trainer_config, seed, t_model_hash, t_dataset_hash, t_trainer_hash):
    assert (
        "detach_core" in trainer_config and trainer_config["detach_core"] is True
    ), "When transferring the core, 'detach_core' in the 'trainer_config' must be set to True."

    # get all trained models corresponding to the combination of model, dataset and trainer
    restricted_trained_model_table = (
        trained_model_table.ModelStorage
        & "model_hash = '{}'".format(t_model_hash)
        & "dataset_hash = '{}'".format(t_dataset_hash)
        & "trainer_hash = '{}'".format(t_trainer_hash)
        & "seed = {}".format(seed)
    )
    state_dict = restricted_trained_model_table.fetch1("model_state", download_path="models/")
    # extract the core parameters of the state dict and initialize the new model with it
    core_dict = OrderedDict([(k, v) for k, v in torch.load(state_dict).items() if k[0:5] == "core."])
    model.load_state_dict(core_dict, strict=False)
