import torch
from ocean4dvarnet.models import GradSolver, Lit4dVarNet


class GenericAEPriorCost(torch.nn.Module):
    """
    A prior cost model using bilinear autoencoders.

    Attributes:
        bilin_quad (bool): Whether to use bilinear quadratic terms.
        conv_in (nn.Conv2d): Convolutional layer for input.
        conv_hidden (nn.Conv2d): Convolutional layer for hidden states.
        bilin_1 (nn.Conv2d): Bilinear layer 1.
        bilin_21 (nn.Conv2d): Bilinear layer 2 (part 1).
        bilin_22 (nn.Conv2d): Bilinear layer 2 (part 2).
        conv_out (nn.Conv2d): Convolutional layer for output.
        down (nn.Module): Downsampling layer.
        up (nn.Module): Upsampling layer.
    """

    def __init__(self, model_ae):
        """
        Initialize the BilinAEPriorCost module.

        Args:
            dim_in (int): Number of input dimensions.
            dim_hidden (int): Number of hidden dimensions.
            kernel_size (int, optional): Kernel size for convolutions. Defaults to 3.
            downsamp (int, optional): Downsampling factor. Defaults to None.
            bilin_quad (bool, optional): Whether to use bilinear quadratic terms. Defaults to True.
        """
        super().__init__()

        self.model_ae = model_ae

    def forward_ae(self, x):
        """
        Perform the forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the autoencoder.
        """
        return self.model_ae(x)

    def forward(self, state):
        """
        Compute the prior cost using the autoencoder.

        Args:
            state (torch.Tensor): The current state tensor.

        Returns:
            torch.Tensor: The computed prior cost.
        """
        return torch.nn.functional.mse_loss(state, self.forward_ae(state))


class GradModelWithCondition(torch.nn.Module):
    """
    A generic conditional model for gradient modulation.

    Attributes:
        grad_model : grad update model
    """

    def __init__(self, grad_model=False, dropout=0.,use_grad_norm=True):
        """
        Initialize the ConvLstmGradModel.

        Args:
            grad_model : grad update model
        """
        super().__init__()
        self.grad_model = grad_model
        self.dropout = torch.nn.Dropout(dropout)
        self.use_grad_norm = use_grad_norm

        if hasattr(self.grad_model, 'dim_3d') == True:
            self.dim_3d = self.grad_model.dim_3d

        if hasattr(self.grad_model, 'dims') == True:
            if self.grad_model.dims == 3:
                self.dim_3d = True

    def reset_state(self, inp):
        """
        Reset the internal state of the LSTM.

        Args:
            inp (torch.Tensor): Input tensor to determine state size.
        """
        self._grad_norm = None


    def forward(self, x, timesteps=None, extra=[]):
        """
        Perform the forward pass of the LSTM.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        if self._grad_norm is None:
            if self.use_grad_norm:
                self._grad_norm = (x**2).mean().sqrt()
            else:
                self._grad_norm = 1.

        x = x / self._grad_norm

        x = self.dropout(x)
        out = self.grad_model.predict(x, timesteps=timesteps, extra=extra)

        return out


class GradSolver_withStep(GradSolver):

    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, lbd=1.0, **kwargs):
        """
        Initialize the GradSolver.

        Args:
            prior_cost (nn.Module): The prior cost function.
            obs_cost (nn.Module): The observation cost function.
            grad_mod (nn.Module): The gradient modulation model.
            n_step (int): Number of optimization steps.
            lr_grad (float, optional): Learning rate for gradient updates. Defaults to 0.2.
            lbd (float, optional): Regularization parameter. Defaults to 1.0.
        """
        self.input_grad_update = kwargs.pop(
            "input_grad_update",
            kwargs["input_grad_update"],
        )
        print("Input type for GradSolver =",self.input_grad_update,flush=True)
        std_init = kwargs.pop("std_init",None)
        print("Std init for GradSolver =",std_init,flush=True)

        super().__init__(prior_cost, obs_cost, grad_mod, n_step=n_step, lr_grad=lr_grad, lbd=lbd,**kwargs)

        self.grad_mod._grad_norm = None
        self.h_state = None

        if std_init is not None:
            self.std_init = std_init

    def init_state(self, batch, x_init=None):
        """
        Initialize the state for optimization.

        Args:
            batch (dict): Input batch containing data.
            x_init (torch.Tensor, optional): Initial state. Defaults to None.
        Returns:
            torch.Tensor: Initialized state.
        """
        if x_init is not None:
            return x_init.detach().requires_grad_(True)

        if hasattr(self, 'std_init') is True :
            x0 = self.std_init * torch.randn_like(batch.input)
            return x0.detach().requires_grad_(True)
        else:
            return torch.zeros_like(batch.input).detach().requires_grad_(True)

    def solver_step(self, state, batch, step, alpha_step=1.):
        """
        Perform a single optimization step.

        Args:
            state (torch.Tensor): Current state.
            batch (dict): Input batch containing data.
            step (float): Current optimization step between 0 and 1.
            alpha_step (float): scaling factor for the step.

        Returns:
            torch.Tensor: Updated state.
        """
        if isinstance(step, float):
            t = torch.tensor([step], device=state.device).repeat(state.shape[0])
        else:
            t = step

        if 'grad' in self.input_grad_update:
            var_cost = (
                self.prior_cost(state)
                + self.lbd**2
                * self.obs_cost(state, batch)
            )
            grad = torch.autograd.grad(var_cost, state)[0]
        else:
            raise Exception("GradSolver_withStep requires 'grad' in input_grad_update")

        gmod = self.grad_mod(grad, timesteps=t, extra=None)

        if hasattr(self.grad_mod, 'dim_3d'):
            if self.grad_mod.dim_3d:
                gmod = gmod.squeeze(1)

        state_update = alpha_step * gmod
        if 'grad' in self.input_grad_update and self.lr_grad > 0.:
            state_update += (
                self.lr_grad * (step + 1) / self.n_step
                * grad[:,:state.shape[1],:,:]
            )

        return state - state_update

    def forward(self, batch, x_init=None, h_state=None,phase='test'):
        """
        Perform the forward pass of the solver.

        Args:
            batch (dict): Input batch containing data.

        Returns:
            torch.Tensor: Final optimized state.
        """
        with torch.set_grad_enabled(True):
            state = self.init_state(batch, x_init=x_init)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(
                    state,
                    batch,
                    step=step/self.n_step,
                    alpha_step=1/self.n_step,
                )
                if (not self.training) and ('grad' in self.input_grad_update):
                    state = state.detach().requires_grad_(True)

        return state


class Lit4dVarNetIgnoreNaN(Lit4dVarNet):
    def __init__(
        self,
        w_mse,w_grad_mse, w_mse_lr, w_grad_mse_lr, w_prior, no_prior=False,
        *args, **kwargs,
    ):
        _val_rec_weight = kwargs.pop(
            "val_rec_weight",
            kwargs["rec_weight"],
        )

        self.osse_with_interp_error = kwargs.pop("osse_with_interp_error")
        self.no_prior = no_prior

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

