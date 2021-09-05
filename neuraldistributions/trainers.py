from copy import deepcopy
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm, trange

from neuralpredictors.training import LongCycler

from .utility import (
    EarlyStopping,
    set_random_seed,
)

from .utility import Correlation


class Trainer:
    def __init__(
        self,
        model,
        dataloaders,
        seed,
        lr=0.005,
        epochs=20,
        use_avg_loss=False,
        loss_accum_batch_n=None,
        early_stopping=True,
        device=torch.device("cuda"),
        measure_for_scheduling="correlation",
        switch_measure_for_scheduling=False,
        compute_conditional_corr=False,
        compute_correlation=True,
        cb=None,
        **kwargs,
    ):

        self.model = model
        self.seed = seed
        self.train_loader, self.val_loader = (
            dataloaders["train"],
            dataloaders["validation"],
        )

        self.lr = lr
        self.epochs = int(epochs)
        self.use_avg_loss = use_avg_loss
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_losses, self.val_losses = [], []
        self.train_corrs_mean, self.val_corrs_mean = [], []
        self.lrs = []
        self.current_epoch = 0

        self.optim_step_count = (
            len(self.train_loader.keys())
            if loss_accum_batch_n is None
            else loss_accum_batch_n
        )
        self.compute_correlation = compute_correlation
        self.measure_for_scheduling = (
            measure_for_scheduling if compute_correlation else "loss"
        )
        self.switch_measure_for_scheduling = (
            switch_measure_for_scheduling if compute_correlation else False
        )
        self.compute_conditional_corr = compute_conditional_corr
        self.cb = cb

        if measure_for_scheduling == "correlation":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                "max",
                factor=0.3,
                patience=10,
                min_lr=1e-8,
                verbose=True,
                threshold_mode="abs",
            )
            self.early_stopping = (
                EarlyStopping(mode="max", patience=20, verbose=True)
                if early_stopping
                else None
            )
        elif measure_for_scheduling == "loss":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                "min",
                factor=0.3,
                patience=10,
                min_lr=1e-8,
                verbose=True,
                threshold_mode="abs",
            )
            self.early_stopping = (
                EarlyStopping(mode="min", patience=20, verbose=True)
                if early_stopping
                else None
            )

    def train_step(self, batch, data_key):

        x, y = batch[:2]
        if self.use_avg_loss:
            return self.model.loss(
                *batch, data_key=data_key, use_avg=self.use_avg_loss
            ) + self.model.regularizer(data_key)

        else:
            dd = self.train_loader[data_key].dataset
            m = dd.images.shape[0] if hasattr(dd, "images") else dd.tensors[0].shape[0]
            k = x.shape[0]
            return (
                np.sqrt(m / k)
                * self.model.loss(*batch, data_key=data_key, use_avg=self.use_avg_loss)
                + self.model.regularizer(data_key)
            ) / (y.shape[0] * y.shape[1])

    def train_one_epoch(self):
        """Training function for a single epoch.

        Returns:
            float: Loss (of the first iteration).

        """
        self.model.train()
        losses = []
        losses_ = 0.0
        self.optimizer.zero_grad()
        for batch_idx, (data_key, batch) in enumerate(
            tqdm(LongCycler(self.train_loader))
        ):

            loss = self.train_step(batch, data_key=data_key)
            loss.backward()

            losses_ += loss.item()
            if (batch_idx + 1) % self.optim_step_count == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(losses_ / self.optim_step_count)
                losses_ = 0

        return losses[0]  # return the first computed loss in the epoch

    def validation_step(self, batch, data_key):

        x, y = batch[:2]
        if self.use_avg_loss:
            return self.model.loss(*batch, data_key=data_key, use_avg=self.use_avg_loss)

        else:
            dd = self.val_loader[data_key].dataset
            m = dd.images.shape[0] if hasattr(dd, "images") else dd.tensors[0].shape[0]
            k = x.shape[0]
            return (
                np.sqrt(m / k)
                * self.model.loss(*batch, data_key=data_key, use_avg=self.use_avg_loss)
                / (y.shape[0] * y.shape[1])
            )

    @torch.no_grad()
    def validate_one_epoch(self):
        """Validation performance.

        Returns:
            float: Loss (of the first iteration).

        """
        self.model.eval()
        losses = []
        losses_ = 0
        for batch_idx, (data_key, batch) in enumerate(LongCycler(self.val_loader)):
            loss = self.validation_step(batch, data_key)
            losses_ += loss.item()

            if (batch_idx + 1) % self.optim_step_count == 0:
                losses.append(losses_ / self.optim_step_count)
                losses_ = 0

        return losses[0]

    def train(self):

        # set the manual seed before running the epochs
        set_random_seed(self.seed)

        # establish a baseline via the validastion dataset
        with torch.no_grad():
            val_loss = self.validate_one_epoch()
            if self.compute_correlation:
                _, val_corrs = Correlation.single_trial(self.val_loader, self.model)
                val_corr_mean = val_corrs.mean()
        if self.early_stopping:
            if self.measure_for_scheduling == "correlation":
                self.early_stopping(val_corr_mean, self.model)
            elif self.measure_for_scheduling == "loss":
                self.early_stopping(val_loss, self.model)

        old_lr = self.lr
        lrs_counter = 0
        for self.current_epoch in range(self.epochs):

            if self.cb is not None:
                self.cb()

            # keep track of the learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr - old_lr:
                self.retrive_best_model()

                # switch the early stopping and lr scheduler to depend on loss (-loglikelihood)
                if (lrs_counter == 0) and self.switch_measure_for_scheduling:
                    self.measure_for_scheduling = "loss"
                    print(
                        f"Changing the measure for early stopping and lr scheduler to {self.measure_for_scheduling}",
                        flush=True,
                    )
                    with torch.no_grad():
                        val_loss = self.validate_one_epoch()

                    self.early_stopping.mode = "min"
                    self.early_stopping.best_score = val_loss
                    self.early_stopping.last_best_score = val_loss
                    self.scheduler.mode = "min"
                    self.scheduler.best = val_loss

                if hasattr(self.model, "apply_changes_while_training") and (
                    lrs_counter == 0
                ):
                    print("Applying changes..", flush=True)
                    self.model.apply_changes_while_training()
                    self.optimizer.param_groups[0]["lr"] = old_lr

                # replace optimizer
                if hasattr(self.model, "change_optimizer") and (lrs_counter == 0):
                    self.optimizer = self.model.change_optimizer(self.optimizer)

                lrs_counter += 1
                old_lr = current_lr

            train_loss = self.train_one_epoch()
            self.train_losses.append(train_loss)

            val_loss = self.validate_one_epoch()
            self.val_losses.append(val_loss)

            if self.compute_correlation:
                with torch.no_grad():

                    _, train_corrs = Correlation.single_trial(
                        self.train_loader, self.model
                    )
                    train_corr_mean = train_corrs.mean()

                    if self.compute_conditional_corr and (lrs_counter > 0):
                        _, val_corrs = Correlation.conditional_single_trial(
                            self.val_loader, self.model
                        )
                    else:
                        _, val_corrs = Correlation.single_trial(
                            self.val_loader, self.model
                        )
                    val_corr_mean = val_corrs.mean()

                self.train_corrs_mean.append(train_corr_mean)
                self.val_corrs_mean.append(val_corr_mean)

            self.lrs.append(self.optimizer.state_dict()["param_groups"][0]["lr"])

            if self.compute_correlation:
                print_string = "Epoch {}/{} | train loss: {:.6f} | val loss: {:.6f} | train corr: {:.6f} | val corr: {:.6f}"
                msg = print_string.format(
                    self.current_epoch + 1,
                    self.epochs,
                    train_loss,
                    val_loss,
                    train_corr_mean,
                    val_corr_mean,
                )
            else:
                print_string = "Epoch {}/{} | train loss: {:.6f} | val loss: {:.6f}"
                msg = print_string.format(
                    self.current_epoch + 1,
                    self.epochs,
                    train_loss,
                    val_loss,
                )

            print(msg, flush=True)

            if self.early_stopping:
                if self.measure_for_scheduling == "correlation":
                    self.early_stopping(val_corr_mean, self.model)
                elif self.measure_for_scheduling == "loss":
                    self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    best_val = (
                        np.array(self.val_corrs_mean).max()
                        if self.measure_for_scheduling == "correlation"
                        else np.array(self.val_losses).min()
                    )
                    msg = "Early stopping at epoch {}. Best val {}: {:.3f}".format(
                        self.current_epoch,
                        self.measure_for_scheduling,
                        best_val,
                    )
                    print(msg, flush=True)
                    break

            if self.measure_for_scheduling == "correlation":
                self.scheduler.step(val_corr_mean)
            elif self.measure_for_scheduling == "loss":
                self.scheduler.step(val_loss)

        if self.early_stopping:
            self.retrive_best_model()

        if self.compute_correlation:
            return {
                "model": self.model,
                "train_losses": np.array(self.train_losses),
                "val_losses": np.array(self.val_losses),
                "train_corrs": np.array(self.train_corrs_mean),
                "val_corrs": np.array(self.val_corrs_mean),
                "lrs": np.array(self.lrs),
            }
        else:
            return {
                "model": self.model,
                "train_losses": np.array(self.train_losses),
                "val_losses": np.array(self.val_losses),
                "lrs": np.array(self.lrs),
            }

    def retrive_best_model(self):
        if self.early_stopping:
            if self.early_stopping.best_model_state_dict is not None:
                print("Retrieve best model..", flush=True)
                self.model.load_state_dict(
                    deepcopy(self.early_stopping.best_model_state_dict)
                )
            else:
                print("Keep existing model..", flush=True)


def base_trainer(
    model,
    dataloaders,
    seed,
    lr=0.005,
    epochs=20,
    use_avg_loss=False,
    loss_accum_batch_n=None,
    early_stopping=True,
    device="cuda",
    measure_for_scheduling="loss",
    switch_measure_for_scheduling=False,
    compute_conditional_corr=False,
    compute_correlation=False,
    **kwargs,
):

    trainer = Trainer(
        model,
        dataloaders,
        seed,
        lr=lr,
        epochs=epochs,
        use_avg_loss=use_avg_loss,
        loss_accum_batch_n=loss_accum_batch_n,
        early_stopping=early_stopping,
        device=device,
        measure_for_scheduling=measure_for_scheduling,
        switch_measure_for_scheduling=switch_measure_for_scheduling,
        compute_conditional_corr=compute_conditional_corr,
        compute_correlation=compute_correlation,
        **kwargs,
    )

    out = trainer.train()

    if compute_correlation:
        score = np.max(out["val_corrs"])
        other_outputs = (
            out["train_losses"],
            out["val_losses"],
            out["train_corrs"],
            out["val_corrs"],
            out["lrs"],
        )
        model_state_dict = out["model"].state_dict()
    else:
        score = np.min(out["val_losses"])
        other_outputs = (out["train_losses"], out["val_losses"], out["lrs"])
        model_state_dict = out["model"].state_dict()

    return score, other_outputs, model_state_dict


class DensityEstimatorTrainer:
    def __init__(
        self,
        model,
        dataloaders,
        seed,
        lr=0.005,
        epochs=20,
        device=torch.device("cuda"),
        cb=None,
        **kwargs,
    ):

        self.model = model
        self.trainloaders = dataloaders["train"]
        self.seed = seed
        self.lr = lr
        self.epochs = int(epochs)
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=0.3,
            patience=10,
            min_lr=1e-8,
            verbose=True,
            threshold_mode="abs",
        )
        self.early_stopping = EarlyStopping(mode="min", patience=20, verbose=False)

        self.losses = []
        self.lrs = []
        self.current_epoch = 0
        self.cb = cb

    def train_one_epoch(self):
        self.model.train()
        losses = []
        self.optimizer.zero_grad()
        for batch in self.trainloaders:
            loss = self.model.loss(*batch) + self.model.regularizer()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses.append(loss.item())

        return np.mean(losses)

    def train(self):
        set_random_seed(self.seed)

        old_lr = self.lr
        lrs_counter = 0
        pbar = trange(self.epochs, desc="Loss: {}".format(np.nan), leave=True)
        for self.current_epoch in pbar:

            if self.cb is not None:
                self.cb()

            # keep track of the learning rate and change stuff when it drop for the first time
            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr - old_lr:
                self.retrive_best_model()
                old_lr = current_lr

                if hasattr(self.model, "apply_changes_while_training") and (
                    lrs_counter == 0
                ):
                    print("Applying changes..", flush=True)
                    self.model.apply_changes_while_training()

                lrs_counter += 1

            loss = self.train_one_epoch()
            self.losses.append(loss)

            self.lrs.append(self.optimizer.state_dict()["param_groups"][0]["lr"])

            print_string = "Epoch {}/{} | loss: {:.2f}"
            msg = print_string.format(
                self.current_epoch + 1, self.epochs, self.losses[-1]
            )
            pbar.set_description(msg)

            self.early_stopping(self.losses[-1], self.model)
            if self.early_stopping.early_stop:
                best_loss = np.array(self.losses).min()
                msg = "Early stopping at epoch {}. Best loss: {:.3f}".format(
                    self.current_epoch, best_loss
                )
                print(msg, flush=True)
                break

            self.scheduler.step(self.losses[-1])

        self.retrive_best_model()

        return (self.model, np.array(self.losses), np.array(self.lrs))

    def retrive_best_model(self):
        if self.early_stopping.best_model_state_dict is not None:
            print("Retrieve best model..", flush=True)
            self.model.load_state_dict(
                deepcopy(self.early_stopping.best_model_state_dict)
            )
        else:
            print("Keep existing model..", flush=True)


def density_estimator_trainer(
    model,
    dataloaders,
    seed,
    lr=0.005,
    epochs=20,
    device=torch.device("cuda"),
    cb=None,
    **kwargs,
):

    trainer = DensityEstimatorTrainer(
        model, dataloaders, seed, lr=lr, epochs=epochs, device=device, cb=cb, **kwargs
    )

    out = trainer.train()

    return out[1].min(), out[1:], out[0].state_dict()