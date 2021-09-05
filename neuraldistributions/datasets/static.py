import os
import zipfile
import warnings
import re
from datetime import datetime
from collections import namedtuple
from tqdm import tqdm
import numpy as np
import torch

from neuralpredictors.data.datasets import StaticImageSet, FileTreeDataset
from neuralpredictors.data.transforms import Subsample, ToTensor

from ..utility import get_dataloader, set_random_seed

extract_data_key = lambda path: "-".join((re.findall(r"\d+", path)[:3] + ["0"]))


def unzip(zipfile_path, extract_to=None):
    extract_to = (
        extract_to if extract_to is not None else zipfile_path.split("static")[0]
    )
    with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def process_dataset_path(path):

    if ".zip" in path:
        if not os.path.exists(path.split(".zip")[0]):
            unzip(path)

        path = path.split(".zip")[0]
        file_tree = True

    elif ".h5" in path:
        file_tree = False

    else:
        file_tree = True

    return path, file_tree


def standardize_based_on(arr, ref_arr, axis=0):
    mean, std = ref_arr.mean(axis=axis, keepdims=True), ref_arr.std(
        axis=0, ddof=1, keepdims=True
    )
    return (arr - mean) / std


def std_normalize_based_on(arr, ref_arr, axis=0):
    std = ref_arr.std(axis=0, ddof=1, keepdims=True)
    return arr / std


def normalize_based_on(arr, ref_arr=None, lower=0.0, upper=1.0, axis=0):
    ref_arr = arr.copy() if ref_arr is None else ref_arr.copy()
    arr = arr.copy()

    min_val = ref_arr.min(axis=axis, keepdims=True)
    ref_arr = ref_arr - min_val
    arr = arr - min_val

    max_val = ref_arr.max(axis=axis, keepdims=True)
    ref_arr = ref_arr / max_val
    arr = arr / max_val

    bound_diff = upper - lower
    ref_arr = ref_arr * bound_diff + lower
    arr = arr * bound_diff + lower

    return arr


def get_ts_in_seconds(ts_str):
    datetime_str = ts_str.split("('")[1].split("')")[0]
    datetime_object = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    return (
        datetime_object.hour * (60 ** 2)
        + datetime_object.minute * 60
        + datetime_object.second
    )


def get_trial_timestamp_in_seconds(trial_timestamp):
    return np.array(list(map(get_ts_in_seconds, trial_timestamp)))


def remove_element(arr, ind):
    to_keep = np.array([True] * len(arr))
    to_keep[ind] = False
    return arr[to_keep]


try:
    from dataport.bcm.static import fetch_non_existing_data
except ImportError:

    def fetch_non_existing_data(func):
        return func

    print("dataport not available, will only be able to load data locally")


@fetch_non_existing_data
def get_dataset(path):
    # pre-process dataset path
    path, file_tree = process_dataset_path(path)

    dat = (
        FileTreeDataset(path, "images", "responses", "behavior", "pupil_center")
        if file_tree
        else StaticImageSet(path, "images", "responses")
    )

    return dat


@fetch_non_existing_data
def get_real_data(
    path,
    seed,
    batch_size,
    normalize_images=True,
    area=None,
    layer="L2/3",
    neuron_ids=None,
    neurons_n=None,
    normalize_neurons=True,
    exclude_neuron_ids=None,
    include_behavior=False,
    return_behavior=False,
    normalize_behavior=True,
    include_pupil_center=False,
    return_pupil_center=False,
    normalize_pupil_center=True,
    include_trial_timestamp=False,
    return_trial_timestamp=False,
    normalize_trial_timestamp=False,
    include_previous_image=False,
    return_previous_response=False,
    include_pixelcoord=False,
    shuffle_train=True,
    return_key=False,
    return_more=False,
    device="cuda",
):

    # pre-process dataset path
    path, file_tree = process_dataset_path(path)

    # set the random seed
    set_random_seed(seed)

    # check arguments are consistent
    if (neuron_ids is not None) and (neurons_n is not None):
        warnings.warn(
            "Both neuron_ids and neurons_n have been assigned a value. neurons_n will be ignored."
        )
        neurons_n = None

    # Create the dataset objects
    dat = (
        FileTreeDataset(path, "images", "responses", "behavior", "pupil_center")
        if file_tree
        else StaticImageSet(path, "images", "responses")
    )

    #### Get the required data (images, responses, etc.) #########################################################

    images = (
        np.expand_dims(np.vstack([d[0] for d in tqdm(dat)]), 1)
        if file_tree
        else dat.images.copy()
    )

    responses = (
        np.vstack([d[1] for d in tqdm(dat)]) if file_tree else dat.responses.copy()
    )
    behavior = (
        np.vstack([d[2] for d in tqdm(dat)]) if file_tree else dat.behavior.copy()
    )
    pupil_center = (
        np.vstack([d[3] for d in tqdm(dat)]) if file_tree else dat.pupil_center.copy()
    )
    tiers = dat.trial_info.tiers.copy() if file_tree else dat.tiers.copy()
    trial_timestamp = get_trial_timestamp_in_seconds(
        dat.trial_info.frame_trial_ts if file_tree else dat.info.frame_trial_ts
    )
    trial_idx = (
        dat.trial_info.trial_idx.copy() if file_tree else dat.info.trial_idx.copy()
    )
    image_ids = (
        dat.trial_info.frame_image_id.copy()
        if file_tree
        else dat.info.frame_image_id.copy()
    )

    # make negative responses 0
    responses[responses < 0.0] = 0.0

    # is area is not specified take all of them
    area = area if area is not None else np.unique(dat.neurons.area)

    #### Sub-select neurons ######################################################################################

    # Set the neuron ids (commonly referred to as "unit ids" in the datasets)
    neuron_ids = neuron_ids if neuron_ids is not None else dat.neurons.unit_ids
    neuron_ids_cond = (
        ~np.isin(neuron_ids, exclude_neuron_ids)
        if exclude_neuron_ids is not None
        else np.isin(dat.neurons.unit_ids, neuron_ids)
    )

    # generate indices for neurons (add conditions for subsampling neurons)
    neuron_cond = neuron_ids_cond & (
        np.isin(dat.neurons.area, area)
        & (dat.neurons.layer == layer)
        & (np.isin(dat.neurons.unit_ids, neuron_ids))
    )

    # get neuron indices based on the combination of conditions
    neurons_idx = np.where(neuron_cond)[0]

    # sample specific number of neurons (per area)
    if neurons_n:

        neurons_n = [neurons_n] if isinstance(neurons_n, int) else neurons_n
        area = [area] if isinstance(area, str) else area
        if len(neurons_n) == len(area):
            neurons_idx = np.hstack(
                [
                    np.random.choice(
                        neurons_idx[dat.neurons.area[neurons_idx] == a],
                        size=neurons_n[idx],
                        replace=False,
                    )
                    for idx, a in enumerate(area)
                ]
            )
        elif len(neurons_n) == 1:
            neurons_idx = np.random.choice(neurons_idx, size=neurons_n, replace=False)
        else:
            raise ValueError("neurons_n cannot be more than the number of areas")

    responses = responses[:, neurons_idx]

    #### Get train val and test indices ################################################################################

    train_idx = tiers == "train"
    val_idx = tiers == "validation"
    test_idx = tiers == "test"

    #### Normalize stuff ###############################################################################################

    if normalize_behavior:
        behavior = std_normalize_based_on(behavior, behavior[train_idx])

    if normalize_images:
        images = images / 255

    if normalize_pupil_center:
        pupil_center = standardize_based_on(
            pupil_center, pupil_center[train_idx], axis=0
        )

    if normalize_neurons:
        responses = std_normalize_based_on(responses, responses[train_idx])

    # normalize trial timestamp
    trial_timestamp = (
        trial_timestamp - trial_timestamp.min()
    )  # set the initial trial to 0
    trial_timestamp = trial_timestamp / 100
    if normalize_trial_timestamp:
        trial_timestamp = trial_timestamp / trial_timestamp.max()

    #### Get previous time point info (images and/or responses) ########################################################

    sorted_ind = np.argsort(trial_idx)
    original_ind = np.argsort(np.argsort(trial_idx))
    first_trial_ind = sorted_ind[0]

    if include_previous_image:
        previous_images = np.concatenate(
            (np.zeros((1, *images.shape[1:])), images[sorted_ind][:-1]), axis=0
        )[original_ind]
        images = np.concatenate((previous_images, images), axis=1)
        tiers[first_trial_ind] = "not_included"

    if return_previous_response:
        previous_responses = np.concatenate(
            (np.zeros((1, responses.shape[1])), responses[sorted_ind][:-1]), axis=0
        )[original_ind]
        tiers[first_trial_ind] = "not_included" if not include_previous_image else tiers

    # update the train, val, and test idx just in case the tiers were updated because of including previous time point stuff
    train_idx = tiers == "train"
    val_idx = tiers == "validation"
    test_idx = tiers == "test"

    #### Include stuff in the image as channels #########################################################################

    if include_behavior:
        behavioral_images = np.ones(
            (images.shape[0], 1, *images.shape[-2:])
        ) * np.expand_dims(behavior, axis=(2, 3))
        images = np.concatenate((images, behavioral_images), axis=1)

    if include_pupil_center:
        pupil_center_images = np.ones(
            (images.shape[0], 1, *images.shape[-2:])
        ) * np.expand_dims(pupil_center, axis=(2, 3))
        images = np.concatenate((images, pupil_center_images), axis=1)

    if include_pixelcoord:
        _, _, h, w = images.shape
        x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        pixelcoords_images = np.repeat(
            np.expand_dims(np.stack((x, y)), 0), images.shape[0], 0
        )
        images = np.concatenate((images, pixelcoords_images), axis=1)

    if include_trial_timestamp:
        trial_timestamp_images = np.ones(
            (images.shape[0], 1, *images.shape[-2:])
        ) * np.expand_dims(trial_timestamp.reshape(-1, 1), axis=(2, 3))
        images = np.concatenate((images, trial_timestamp_images), axis=1)

    #### Choose what to add in the dataloaders ###########################################################################

    train_arrays = [images[train_idx], responses[train_idx]]
    val_arrays = [images[val_idx], responses[val_idx]]
    test_arrays = [images[test_idx], responses[test_idx]]
    names = ["inputs", "targets"]

    if return_previous_response:
        train_arrays.append(previous_responses[train_idx])
        val_arrays.append(previous_responses[val_idx])
        test_arrays.append(previous_responses[test_idx])
        names.append("prev_targets")

    if return_behavior:
        train_arrays.append(behavior[train_idx])
        val_arrays.append(behavior[val_idx])
        test_arrays.append(behavior[test_idx])
        names.append("behavior")

    if return_pupil_center:
        train_arrays.append(pupil_center[train_idx])
        val_arrays.append(pupil_center[val_idx])
        test_arrays.append(pupil_center[test_idx])
        names.append("pupil_center")

    if return_trial_timestamp:
        train_arrays.append(trial_timestamp[train_idx])
        val_arrays.append(trial_timestamp[val_idx])
        test_arrays.append(trial_timestamp[test_idx])
        names.append("trial_timestamp")

    #### Create the dataloaders #########################################################################################

    train_loader = get_dataloader(
        *train_arrays,
        names=names,
        batch_size=int(batch_size),
        shuffle=shuffle_train,
        device=device
    )
    val_loader = get_dataloader(
        *val_arrays,
        names=names,
        batch_size=int(batch_size),
        shuffle=False,
        device=device
    )
    test_loader = get_dataloader(
        *test_arrays,
        names=names,
        batch_size=int(batch_size),
        shuffle=False,
        device=device
    )

    # mean response of the neurons
    mean_responses = None  # responses[neurons_idx].mean(0).reshape(1,-1) if file_tree else dat.transformed_mean().responses[neurons_idx].reshape(1, -1)

    if hasattr(dat.neurons, "cell_motor_coordinates"):
        neuron_locs = dat.neurons.cell_motor_coordinates[neurons_idx]
    else:
        neuron_locs = None

    neurons = namedtuple(
        "neurons", ["cell_motor_coordinates", "ids", "areas", "animal_ids"]
    )
    neurons.cell_motor_coordinates = neuron_locs
    neurons.ids = neuron_ids[neurons_idx]
    neurons.areas = dat.neurons.area[neurons_idx]
    neurons.animal_ids = dat.neurons.animal_ids[neurons_idx]

    trial_info = {
        tier: namedtuple("trial_info", ["timestamps", "image_ids"])
        for tier in ["train", "validation", "test"]
    }
    trial_info["train"].timestamps = trial_timestamp[train_idx]
    trial_info["train"].image_ids = image_ids[train_idx]
    trial_info["validation"].timestamps = trial_timestamp[val_idx]
    trial_info["validation"].image_ids = image_ids[val_idx]
    trial_info["test"].timestamps = trial_timestamp[test_idx]
    trial_info["test"].image_ids = image_ids[test_idx]

    train_loader.dataset.neurons = neurons
    train_loader.dataset.trial_info = trial_info["train"]
    val_loader.dataset.neurons = neurons
    val_loader.dataset.trial_info = trial_info["validation"]
    test_loader.dataset.neurons = neurons
    test_loader.dataset.trial_info = trial_info["test"]

    #### Return a dictionary containing the dataloaders #################################################################
    out = dict(train=train_loader, validation=val_loader, test=test_loader)
    if return_more:
        out.update(
            dict(
                neuron_ids=neuron_ids[neurons_idx],
                areas=area,
                area_ids=dat.neurons.area[neurons_idx],
                mean_responses=mean_responses,
                trial_timestamp=dict(
                    train=trial_timestamp[train_idx],
                    validation=trial_timestamp[val_idx],
                    test=trial_timestamp[test_idx],
                ),
                image_ids=dict(
                    train=image_ids[train_idx],
                    validation=image_ids[val_idx],
                    test=image_ids[test_idx],
                ),
                neuron_locs=neuron_locs,
            )
        )

    # generate the data_key based on path string
    if return_key:
        # data_key = path.split("static")[-1].split(".")[0].replace("preproc", "")
        data_key = extract_data_key(path)

        return data_key, out

    else:
        return out


def mouse_static_loaders(
    paths,
    seed,
    batch_size,
    normalize_images=True,
    area=None,
    layer="L2/3",
    neuron_ids=None,
    neurons_n=None,
    normalize_neurons=True,
    return_more=True,
    exclude_neuron_ids=None,
    include_behavior=False,
    return_behavior=False,
    normalize_behavior=True,
    include_pupil_center=False,
    return_pupil_center=False,
    normalize_pupil_center=True,
    include_trial_timestamp=False,
    return_trial_timestamp=False,
    normalize_trial_timestamp=False,
    include_previous_image=False,
    return_previous_response=False,
    include_pixelcoord=False,
    shuffle_train=True,
    device="cuda",
):

    loaders = {}
    for path in paths:
        data_key, loader = get_real_data(
            path,
            seed,
            batch_size,
            normalize_images=normalize_images,
            area=area,
            layer=layer,
            neuron_ids=neuron_ids,
            neurons_n=neurons_n,
            device=device,
            normalize_neurons=normalize_neurons,
            return_more=return_more,
            return_key=True,
            exclude_neuron_ids=exclude_neuron_ids,
            include_behavior=include_behavior,
            return_behavior=return_behavior,
            normalize_behavior=normalize_behavior,
            include_trial_timestamp=include_trial_timestamp,
            include_pupil_center=include_pupil_center,
            return_pupil_center=return_pupil_center,
            normalize_pupil_center=normalize_pupil_center,
            return_trial_timestamp=return_trial_timestamp,
            normalize_trial_timestamp=normalize_trial_timestamp,
            return_previous_response=return_previous_response,
            include_previous_image=include_previous_image,
            include_pixelcoord=include_pixelcoord,
            shuffle_train=shuffle_train,
        )
        loaders[data_key] = loader

    keys = list(loaders.values())[0].keys()
    dataloaders = {}
    for key in keys:
        dataloaders[key] = {}
        for data_key, loader in loaders.items():
            dataloaders[key][data_key] = loader[key]

    return dataloaders