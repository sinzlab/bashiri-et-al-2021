from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(
        self,
        mode="min",
        model_path=None,
        patience=7,
        verbose=False,
        delta=0.0,
        update_best_score=True,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        """
        self.mode = mode
        self.model_path = "{}/checkpoint.pt".format(model_path) if model_path else None
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.last_best_score = np.nan
        self.delta = delta
        self.best_model_state_dict = None
        self.update_best_score = update_best_score

    def check_min(self, current_score, model):
        # check if the new score is smaller than the current best score (with some margin)

        if (self.best_score + self.delta) < current_score:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}",
                    flush=True,
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = (
                current_score if self.update_best_score else self.best_score
            )
            self.best_model_state_dict = deepcopy(model.state_dict())
            self.save_checkpoint(current_score, model)
            self.counter = 0

    def check_max(self, current_score, model):
        # check if the new score is bigger than the best score (with some margin)
        if current_score < (self.best_score - self.delta):
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}",
                    flush=True,
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = (
                current_score if self.update_best_score else self.best_score
            )
            self.best_model_state_dict = deepcopy(model.state_dict())
            self.save_checkpoint(current_score, model)
            self.counter = 0

    def __call__(self, current_score, model):

        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(current_score, model)
        else:
            self.check_min(
                current_score, model
            ) if self.mode == "min" else self.check_max(current_score, model)

    def save_checkpoint(self, new_score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Score improved ({self.last_best_score:.6f} --> {new_score:.6f}).",
                flush=True,
            )
        self.last_best_score = new_score if self.update_best_score else self.best_score

        if self.model_path:
            torch.save(model.state_dict(), self.model_path)