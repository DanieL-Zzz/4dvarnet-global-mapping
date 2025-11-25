"""
Add the 4DVarNet core solver, i. e. ...
"""
import torch
from ocean4dvarnet.models import GradSolver

from contrib.glorys12 import Lit4dVarNetIgnoreNaN

class Lit4DVarNetSteppedSolver(Lit4dVarNetIgnoreNaN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_detach = kwargs.get('n_detach', 1)

    def forward(self, batch, state):
        return self.solver(batch, state)

    def base_step(self, batch, phase=None):
        losses = []
        state = None

        for _ in range(self.n_detach):
            out = self(batch, state)
            losses.append(self.weighted_mse(
                out - batch.tgt,
                self.get_rec_weight(phase),
            ))
            state = out.detach().requires_grad_(True)

        if self.training:
            loss = torch.stack(losses).sum()
        else:
            loss = losses[-1]

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


class InitializableSolver(GradSolver):
    @torch.enable_grad()
    def forward(self, batch, state=None):
        if state is None:
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)

        for step in range(self.n_step):
            state = self.solver_step(state, batch, step)

            if not self.training:
                state = state.detach().requires_grad_(True)

        if not self.training:
            state = self.prior_cost.forward_ae(state)

        return state
