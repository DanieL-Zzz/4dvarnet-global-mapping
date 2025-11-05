"""
Adapted by Ronan Fablet
"""
import functools as ft
from collections import namedtuple

import kornia.filters as kfilts
import numpy as np
import torch
from ocean4dvarnet.data import BaseDataModule, TrainingItem
from ocean4dvarnet.models import Lit4dVarNet

TrainingItemwithLonLat = namedtuple(
    "TrainingItemwithLonLat",
    ["input", "tgt", "lon", "lat"],
)


class DistinctNormDataModulewInputTracks(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tgt = self.input_da[0]
        self.inp = self.input_da[1]
        self.input_mask = None

    def norm_stats(self):
        if self._norm_stats is None:
            raise NormParamsNotProvided()
        return self._norm_stats

    def post_fn(self, phase):
        m, s = self.norm_stats()[phase]

        def normalize(item):
            return (item - m) / s

        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                TrainingItemwithLonLat._make,
                lambda item: item._replace(tgt=normalize(item.tgt)),
                lambda item: item._replace(input=normalize(item.input)),
            ],
        )

    def setup(self, stage="test"):
        self.train_ds = LazyXrDatasetwInputTrack(
            (
                self.tgt.sel(self.domains["train"]),
                self.inp.sel(self.domains["train"]),
            ),
            **self.xrds_kw["train"],
            postpro_fn=self.post_fn("train"),
            mask=self.input_mask,
        )

        self.val_ds = LazyXrDatasetwInputTrack(
            (
                self.tgt.sel(self.domains["val"]),
                self.inp.sel(self.domains["val"]),
            ),
            **self.xrds_kw["val"],
            postpro_fn=self.post_fn("val"),
            mask=self.input_mask,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=1,
            num_workers=1,
        )


class LazyXrDatasetwInputTrack(torch.utils.data.Dataset):
    def __init__(
        self,
        ds,
        patch_dims,
        domain_limits=None,
        strides=None,
        postpro_fn=None,
        noise_type=None,
        noise=None,
        noise_spatial_perturb=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.ds = tuple(d.sel(**(domain_limits or {})) for d in ds)
        self.patch_dims = patch_dims
        self.strides = strides or {}
        _dims = ("variable",) + tuple(k for k in self.ds[0].dims)
        _shape = (2,) + tuple(self.ds[0][k].shape[0] for k in self.ds[0].dims)
        ds_dims = dict(zip(_dims, _shape))

        self.ds_size = {
            dim: max(
                (ds_dims[dim] - patch_dims[dim]) // strides.get(dim, 1) + 1,
                0,
            )
            for dim in patch_dims
        }

        self.periodic_dim = kwargs.get('periodic_dim')
        if self.periodic_dim:
            patch_criterion = (
                (ds_dims[self.periodic_dim] - patch_dims[self.periodic_dim])
                % strides[self.periodic_dim]
            )
            if patch_criterion != 0:
                self.ds_size[self.periodic_dim] += 1

        self.padded_dim = kwargs.get('padded_dim')
        if self.padded_dim:
            patch_criterion = (
                (ds_dims[self.padded_dim] - patch_dims[self.padded_dim])
                % strides[self.padded_dim]
            )
            if patch_criterion != 0:
                self.ds_size[self.padded_dim] += 1

        self._rng = np.random.default_rng()
        self.noise = noise
        self.noise_spatial_perturb = noise_spatial_perturb

        if noise_type is not None:
            self.noise_type = noise_type
        else:
            self.noise_type = "uniform-constant"

        self.mask = kwargs.get("mask")

        print(self.noise_type, flush=True)

    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords(self):
        self.return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self.return_coords = False
            return coords

    def __getitem__(self, item):
        sl = {}
        _zip = zip(
            self.ds_size.keys(), np.unravel_index(item, tuple(self.ds_size.values()))
        )

        for dim, idx in _zip:
            sl[dim] = slice(
                self.strides.get(dim, 1) * idx,
                self.strides.get(dim, 1) * idx + self.patch_dims[dim],
            )

        da = [d.isel(**sl) for d in self.ds]

        # Longitude extension
        offset_lon = 0
        if self.periodic_dim:
            if len(da[0].lon) < self.patch_dims['lon']:
                offset_lon = self.patch_dims['lon'] - len(da[0].lon)
                sl['lon'] = slice(
                    sl['lon'].start - offset_lon, sl['lon'].stop - offset_lon,
                )
                da[0] = (
                    self.ds[0]
                    .isel(time=sl['time'])
                    .roll(lon=-offset_lon)
                    .assign_coords(lon=lambda x: x.lon - offset_lon)
                    .sortby('lon')
                    .isel({k: v for k, v in sl.items() if k != 'time'})
                )
                da[1] = (
                    self.ds[1]
                    .isel(time=sl['time'])
                    .roll(lon=-offset_lon)
                    .assign_coords(lon=lambda x: x.lon - offset_lon)
                    .sortby('lon')
                    .isel({k: v for k, v in sl.items() if k != 'time'})
                )

        # Latitude extension
        offset_lat = 0
        if self.padded_dim:
            if len(da[0].lat) < self.patch_dims['lat']:
                offset_lat = self.patch_dims['lat'] - len(da[0].lat)
                da[0] = (
                    da[0]
                    .pad(
                        pad_width=dict(lat=(0, offset_lat)),
                        mode='constant',
                        constant_values=np.nan,
                    )
                )
                da[1] = (
                    da[1]
                    .pad(
                        pad_width=dict(lat=(0, offset_lat)),
                        mode='constant',
                        constant_values=np.nan,
                    )
                )

        data_tgt = da[0].data.astype(np.float32).squeeze()
        data_input = da[1].data.astype(np.float32).squeeze()

        if self.return_coords:
            return da[0].coords.to_dataset()[list(self.patch_dims)]

        if self.noise is not None:
            if self.noise_type == "uniform-constant":
                noise = self._rng.uniform(
                    -self.noise, self.noise, data_input.shape
                ).astype(np.float32)
                data_input = data_input + noise
            elif self.noise_type == "gaussian+uniform":
                scale = self._rng.uniform(0.0, self.noise, 1).astype(np.float32)
                noise = self._rng.normal(0.0, 1.0, data_input.shape).astype(np.float32)
                data_input = data_input + scale * noise

        item = TrainingItemwithLonLat(
            data_input,
            data_tgt,
            da[0].coords["lon"].data.astype(np.float32),
            da[0].coords["lat"].data.astype(np.float32),
        )

        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        return item


class Lit4dVarNetIgnoreNaN(Lit4dVarNet):
    def __init__(
        self,
        w_mse,
        w_grad_mse,
        w_mse_lr,
        w_grad_mse_lr,
        w_prior,
        *args,
        **kwargs,
    ):
        _val_rec_weight = kwargs.pop("val_rec_weight", kwargs["rec_weight"])

        self.osse_with_interp_error = kwargs.pop("osse_with_interp_error", False)

        super().__init__(*args, **kwargs)

        self.register_buffer(
            "val_rec_weight",
            torch.from_numpy(_val_rec_weight),
            persistent=False,
        )

        self._n_rejected_batches = 0

        self.w_mse = w_mse
        self.w_grad_mse = w_grad_mse
        self.w_mse_lr = w_mse_lr
        self.w_grad_mse_lr = w_grad_mse_lr
        self.w_prior = w_prior

    def get_rec_weight(self, phase):
        rec_weight = self.rec_weight
        if phase == "val":
            rec_weight = self.val_rec_weight
        return rec_weight

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        if loss is None:
            self._n_rejected_batches += 1
        return loss

    def on_train_epoch_end(self):
        self.log(
            "n_rejected_batches",
            self._n_rejected_batches,
            on_step=False,
            on_epoch=True,
        )

    def sample_osse_data_with_l3interp_errr(self, batch):
        # to be implemented in child class if needed
        # patch dimensions
        K = batch.input.shape[0]
        N = batch.input.shape[2]
        M = batch.input.shape[3]
        T = batch.input.shape[1]

        # start with time interpolation error only
        dt = 1.0 * (
            torch.rand((K, T - 2, N, M), dtype=torch.float32, device=batch.input.device)
            - 0.5
        )
        dt = torch.nn.functional.avg_pool2d(dt, (4, 4))
        dt = torch.nn.functional.interpolate(dt, scale_factor=4.0, mode="bilinear")

        inp_dt = (dt >= 0.0) * (
            (1.0 - dt) * batch.tgt[:, 1:-1, :, :] + dt * batch.tgt[:, 2:, :, :]
        )
        inp_dt += (dt < 0.0) * (
            (1 + dt) * batch.tgt[:, 1:-1, :, :] - dt * batch.tgt[:, 0:-2, :, :]
        )
        inp_dt = torch.cat(
            (batch.tgt[:, 0:1, :, :], inp_dt, batch.tgt[:, -1:, :, :]), dim=1
        )

        # space interpolation error
        scale_spatial_perturbation = 1.0
        dx = scale_spatial_perturbation * torch.rand(
            (K, T, N, M), dtype=torch.float32, device=batch.input.device
        )
        dy = scale_spatial_perturbation * torch.rand(
            (K, T, N, M), dtype=torch.float32, device=batch.input.device
        )

        dx = torch.nn.functional.avg_pool2d(dx, (4, 4))
        dx = torch.nn.functional.interpolate(dx, scale_factor=4.0, mode="bilinear")

        dy = torch.nn.functional.avg_pool2d(dy, (4, 4))
        dy = torch.nn.functional.interpolate(dy, scale_factor=4.0, mode="bilinear")

        dx = dx[:, :, :-1, :-1]
        dy = dy[:, :, :-1, :-1]

        inp_dxdydt = inp_dt[:, :, :-1, :-1] * (1 - dx) * (1 - dy)
        inp_dxdydt += inp_dt[:, :, :-1, 1:] * (1 - dx) * dy
        inp_dxdydt += inp_dt[:, :, 1:, :-1] * dx * (1 - dy)
        inp_dxdydt += inp_dt[:, :, 1:, 1:] * dx * dy

        inp_dxdydt = torch.where(
            inp_dxdydt.isfinite(), inp_dxdydt, batch.tgt[:, :, :-1, :-1]
        )

        inp_dxdydt = torch.cat((inp_dxdydt, inp_dt[:, :, -1:, :-1]), dim=2)
        inp_dxdydt = torch.cat((inp_dxdydt, inp_dt[:, :, :, -1:]), dim=3)

        input = torch.where(
            batch.input.isfinite(),
            batch.input + inp_dxdydt.detach() - batch.tgt,
            torch.nan,
        )
        input = input.detach()

        return TrainingItem(input, batch.tgt)

    def loss_mse(self, batch, out, phase):
        loss = self.weighted_mse(
            out - batch.tgt,
            self.get_rec_weight(phase),
        )

        grad_loss = self.weighted_mse(
            kfilts.sobel(out) - kfilts.sobel(batch.tgt),
            self.get_rec_weight(phase),
        )

        return loss, grad_loss

    def loss_prior(self, batch, out, phase):
        # prior cost for estimated latent state
        loss_prior_out = self.solver.prior_cost(out)  # Why using init_state

        # prior cost for true state
        loss_prior_tgt = self.solver.prior_cost(batch.tgt.nan_to_num())

        return loss_prior_out, loss_prior_tgt

    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < 0.5:
            return None, None

        # osse input
        if self.osse_with_interp_error and (phase in ["train", "val"]):
            batch_ = self.sample_osse_data_with_l3interp_errr(batch)
        else:
            batch_ = batch

        # apply base-step
        loss, out = self.base_step(batch_, phase)

        loss_mse = self.loss_mse(batch, out, phase)
        loss_prior = self.loss_prior(batch, out.detach(), phase)

        training_loss = self.w_mse * loss_mse[0] + self.w_grad_mse * loss_mse[1]
        training_loss += self.w_prior * loss_prior[0] + self.w_prior * loss_prior[1]

        with torch.no_grad():
            self.log(
                f"{phase}_mse",
                10000 * loss_mse[0] * self.norm_stats[phase][1] ** 2,
                prog_bar=True,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_loss",
                training_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

            self.log(
                f"{phase}_gloss",
                loss_mse[1],
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_ploss_out",
                loss_prior[0],
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_ploss_gt",
                loss_prior[1],
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

        return training_loss, out

    def base_step(self, batch, phase):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.get_rec_weight(phase))

        return loss, out


class LitUnetFromLit4dVarNetIgnoreNaN(Lit4dVarNetIgnoreNaN):
    def step(self, batch, phase):
        if self.training and batch.tgt.isfinite().float().mean() < 0.5:
            return None, None

        _, out = self.base_step(batch, phase)
        loss_mse = self.loss_mse(batch, out, phase)

        self.log(
            f"{phase}_gloss",
            loss_mse[1],
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        training_loss = self.w_mse * loss_mse[0] + self.w_grad_mse * loss_mse[1]

        # log
        self.log(
            f"{phase}_gloss",
            loss_mse[1],
            prog_bar=False,
            on_step=False,
            on_epoch=True,  # sync_dist=True,
        )

        return training_loss, out

    def base_step(self, batch, phase):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.get_rec_weight(phase))

        with torch.no_grad():
            self.log(
                f"{phase}_mse",
                10000 * loss * self.norm_stats[phase][1] ** 2,
                prog_bar=True,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )
            self.log(
                f"{phase}_loss",
                loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,  # sync_dist=True,
            )

        return loss, out


class NormParamsNotProvided(Exception):
    """Normalisation parameters have not been provided"""


class UnetSolver(torch.nn.Module):
    def __init__(self, dim_in, channel_dims, max_depth=None, bias=True):
        super().__init__()

        if max_depth is not None:
            self.max_depth = np.max(max_depth, len(channel_dims) // 3)
        else:
            self.max_depth = len(channel_dims) // 3

        self.ups = torch.nn.ModuleList()
        self.up_pools = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        self.down_pools = torch.nn.ModuleList()
        self.residues = list()

        self.bottom_transform = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3 - 1],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias,
            ),
            torch.nn.ReLU(),
        )

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=dim_in,
                padding="same",
                kernel_size=3,
                bias=bias,
            )
        )

        self.final_linear = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_in))

        for depth in range(self.max_depth):
            self.ups.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 2] * 2,
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    torch.nn.ReLU(),
                )
            )
            self.up_pools.append(
                torch.nn.ConvTranspose2d(
                    in_channels=channel_dims[depth * 3 + 3],
                    out_channels=channel_dims[depth * 3 + 2],
                    kernel_size=2,
                    stride=2,
                    bias=bias,
                )
            )
            self.downs.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=dim_in
                        if depth == 0
                        else channel_dims[depth * 3 - 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3],
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    torch.nn.ReLU(),
                )
            )
            self.down_pools.append(torch.nn.MaxPool2d(kernel_size=2))

    def unet_step(self, x, depth):
        x, residue = self.down(x, depth)
        self.residues.append(residue)

        if depth == self.max_depth - 1:
            x = self.bottom_transform(x)
        else:
            x = self.unet_step(x, depth + 1)

        return self.up(x, depth)

    def forward(self, batch):
        x = batch.input
        x = x.nan_to_num()

        return self.predict(x)

    def predict(self, x):
        x = self.final_up(self.unet_step(x, depth=0))
        x = torch.permute(x, dims=(0, 2, 3, 1))
        x = self.final_linear(x)
        x = torch.permute(x, dims=(0, 3, 1, 2))
        return x

    def down(self, x, depth):
        x = self.downs[depth](x)
        return self.down_pools[depth](x), x

    def up(self, x, depth):
        x = self.up_pools[depth](x)
        x = self.concat_residue(x)
        return self.ups[depth](x)

    def concat_residue(self, x):
        if len(self.residues) != 0:
            residue = self.residues.pop(-1)

            _, _, h_x, w_x = x.shape
            _, _, h_r, w_r = residue.shape

            pad_h = h_r - h_x
            pad_w = w_r - w_x

            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(
                    x, (0, pad_w, 0, pad_h), mode="reflect", value=0
                )

            return torch.concat((x, residue), dim=1)
        else:
            return x


class UnetSolverBilin(UnetSolver):
    def __init__(
        self,
        dim_in,
        channel_dims,
        max_depth=None,
        dim_out=None,
        interp_mode="bilinear",
        dropout=0.1,
        activation_layer=torch.nn.ReLU(),
        bias=True,
    ):
        super().__init__(dim_in, channel_dims, max_depth=max_depth, bias=bias)

        if dim_out is None:
            dim_out = dim_in

        self.up_pools = torch.nn.ModuleList()
        self.down_pools = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()

        self.interp_mode = interp_mode
        self.dropout = dropout

        self.bottom_transform = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3 - 1],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias,
            ),
            activation_layer,
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias,
            ),
            activation_layer,
        )

        for depth in range(self.max_depth):
            self.ups.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 2] * 2,
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                )
            )
            self.up_pools.append(
                UpsampleWInterpolate(
                    channels=channel_dims[depth * 3 + 3],
                    use_conv=True,
                    out_channels=channel_dims[depth * 3 + 2],
                    interp_mode=self.interp_mode,
                )
            )
            self.downs.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=dim_in
                        if depth == 0
                        else channel_dims[depth * 3 - 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3],
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                )
            )

            self.down_pools.append(torch.nn.AvgPool2d(kernel_size=2))

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=4 * dim_out,
                padding="same",
                kernel_size=3,
                bias=bias,
            )
        )
        self.final_linear = torch.nn.Sequential(
            torch.nn.Linear(4 * dim_out, dim_out, bias=bias)
        )


class UpsampleWInterpolate(torch.nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self, channels, use_conv, out_channels=None, interp_mode="bilinear", bias=True
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.interp_mode = interp_mode
        if use_conv:
            self.conv = torch.nn.Conv2d(
                in_channels=channels,
                out_channels=out_channels,
                padding="same",
                kernel_size=1,
                bias=bias,
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode=self.interp_mode)
        if self.use_conv:
            x = self.conv(x)
        return x


def cosanneal_lr_adam_base(lit_mod, lr, T_max=100, weight_decay=0.0):
    """
    Configure an Adam optimizer with cosine annealing learning rate scheduling.

    Args:
        lit_mod: The Lightning module containing the model.
        lr (float): The base learning rate.
        T_max (int): Maximum number of iterations for the scheduler.
        weight_decay (float): Weight decay for the optimizer.

    Returns:
        dict: A dictionary containing the optimizer and scheduler.
    """
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.parameters(), "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }
