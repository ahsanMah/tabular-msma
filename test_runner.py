import matplotlib.pyplot as plt
import ml_collections
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch

import datasets
from models.simple_nn import SimpleNet
from torch.utils.data import DataLoader
from pytorch_lightning.profiler import PyTorchProfiler

from configs import simple_dsm_config as config_builder

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["figure.dpi"] = 110


if __name__ == "__main__":

    # Make sure the current PyTorch binary was built with MPS enabled
    print("MPS built:", torch.backends.mps.is_built())
    # And that the current hardware and MacOS version are sufficient to
    # be able to use MPS
    print("MPS found:", torch.backends.mps.is_available())

    device = torch.device("cpu")
    config = config_builder.get_default_configs()

    # Load data
    X_train, (X_val_in, X_val_out), (X_test_in, X_test_out) = datasets.build_dataset(
        config, val_ratio=0.2
    )
    train_loader = DataLoader(X_train[:1024], batch_size=16, num_workers=5)
    val_in_loader = DataLoader((X_val_in, X_val_out), batch_size=128, num_workers=0)

    model = SimpleNet(config).to(device)

    profiler = PyTorchProfiler(filename="perf-logs")
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=5,
        fast_dev_run=None,
        val_check_interval=0.3,
        profiler=profiler,
    )

    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=val_in_loader
    )

    # model.eval()
    # with torch.no_grad():
    #     id_scores = torch.linalg.norm(model(next(iter(val_in_loader))).reshape(X_val_in.shape[0],-1), dim=-1).cpu()
    #     ood_scores = torch.linalg.norm(model(next(iter(val_ood_loader))).reshape(X_val_out.shape[0],-1), dim=-1).cpu()
    # id_scores.shape, ood_scores.shape

    # df = pd.DataFrame.from_dict({"Inlier":id_scores, "Outlier":ood_scores})
    # df.plot(kind="hist")
    # plt.savefig("test-hist.png")
