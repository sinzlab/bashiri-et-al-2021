from collections import namedtuple
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class NamedTensorDataset(Dataset):
    """
    Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, names=("inputs", "targets")):
        assert all(
            tensors[0].size(0) == tensor.size(0) for tensor in tensors
        ), "tensors have differing sample size."
        assert len(tensors) == len(names), "there should be a name for every tensor"
        self.tensors = tensors
        self.DataPoint = namedtuple("DataPoint", names)

    def __getitem__(self, index):
        return self.DataPoint(*[tensor[index] for tensor in self.tensors])

    def __len__(self):
        return self.tensors[0].size(0)


def get_dataloader(
    *arrays, names=("inputs", "targets"), batch_size=64, shuffle=False, device="cuda"
):
    """Returns data generators the given input and output

    Args:
        imgs (Numpy array): Inputs.
        targets (Numpy array): Outputs.
        batch_size (int): Batch size for each quiery of data from DataLoader object.

    Returns:
        DataLoader object: iterable dataloader object.

    """

    tensors = [torch.from_numpy(t.astype(np.float32)).to(device) for t in arrays]
    name = (
        names
        if len(tensors) == len(names)
        else ["inputs", "targets"] + [f"info_{ind}" for ind in range(len(tensors) - 2)]
    )
    # import ipdb; ipdb.set_trace()
    data_set = NamedTensorDataset(*tensors, names=names)
    data_loader = data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)

    return data_loader


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def imread(filename, grayscale=True, xres=None, yres=None):

    img = Image.open(filename)
    h, w, *_ = np.array(img).shape
    aspectratio = h / w

    xres = xres if xres is not None else w
    yres = yres if yres is not None else int(xres * aspectratio)

    image = np.array(img.resize((xres, yres), Image.ANTIALIAS))

    gray_image = (
        rgb2gray(image)[None, None, :, :].astype(np.float32)
        if grayscale
        else image.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    )

    # make sure the entries are between 0 and 1
    gray_image = gray_image - gray_image.min()
    gray_image = gray_image / gray_image.max()

    return gray_image
