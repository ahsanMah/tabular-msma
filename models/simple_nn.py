import torch
import torch.nn as nn
import pytorch_lightning as pl
from losses import build_loss_fn


class SimpleNet(pl.LightningModule):
    def __init__(self, config, hidden_dims=64) -> None:
        super().__init__()

        input_dims = config.input_dims
        # hidden_dims = config.hidden_dims

        self.seq_modules = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, input_dims),
        )

        self.loss_fn = build_loss_fn(config)
        self.sigma = config.scale
        self.reduce_op = config.reduce_op

        self.apply(self._init_weights)
        self.save_hyperparameters()

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                module.bias.data.zero_()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

    def forward(self, x) -> torch.Tensor:
        x = self.seq_modules(x)
        return x

    def training_step(self, x_batch) -> torch.Tensor:
        z = torch.randn_like(x_batch) * self.sigma
        x_pert = x_batch + z
        scores = self.forward(x_pert)
        loss = self.loss_fn(scores, x_batch, x_pert)

        bs = x_batch.shape[0]
        out_dict = {"loss": loss, "batch_sz": bs}

        return out_dict

    def validation_step(self, val_batch, val_batch_idxs):
        with torch.no_grad():
            id_batch, ood_batch = val_batch
            z = torch.randn_like(id_batch) * self.sigma
            x_pert = id_batch + z
            scores = self.forward(x_pert)
            val_loss = self.loss_fn(scores, id_batch, x_pert)

        bs = id_batch.shape[0]
        out_dict = {"val_loss": val_loss, "batch_sz": bs}

        # If using sum reduction in loss
        # div by batch size

        if self.current_epoch % 2 == 0:
            # Calculate score norms
            id_scores = self.scorer(id_batch)
            ood_scores = self.scorer(ood_batch)

            out_dict["id_scores"] = id_scores
            out_dict["ood_scores"] = ood_scores

        return out_dict

    def scorer(self, x_batch) -> torch.Tensor:

        scores = self.forward(x_batch)
        scores = scores.reshape(x_batch.shape[0], -1)
        score_norms = torch.linalg.norm(scores, dim=1)

        return score_norms

    #### Logging logic here ####

    def training_step_end(self, outputs):

        bs = outputs["batch_sz"]
        loss = outputs["loss"]

        if self.reduce_op == "sum":
            loss /= bs

        self.log("loss/train", loss, prog_bar=True)

        return

    def validation_step_end(self, outputs):

        bs = outputs["batch_sz"]
        val_loss = outputs["val_loss"]

        if self.reduce_op == "sum":
            val_loss /= bs

        self.log("loss/val", val_loss)

        if "id_scores" in outputs:
            tensorboard = self.logger.experiment
            tensorboard.add_histogram(
                "score_norms/ID", outputs["id_scores"], self.global_step
            )
            tensorboard.add_histogram(
                "score_norms/OOD", outputs["ood_scores"], self.global_step
            )

        return
