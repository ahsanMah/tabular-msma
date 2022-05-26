import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit

SEED = 42


def build_dataset(config, val_ratio=0.1):
    """
    Builds dataset splits according to validation ratio

    Args:
        config (_type_): _description_
        val_ratio (float, optional): _description_. Defaults to 0.1.

    Returns:
        tuple: Train set with (1 - val_ratio) samples,
               Validation/Test split with balanced number of inliers/outliers
    """
    if config.dataset == "credit_fraud":

        df = pd.read_csv("raw_data/creditcard.csv", dtype=np.float32).drop(
            columns="Time"
        )
        # df = shuffle(df)

        X = df.drop(columns="Class")
        y = df.Class

        X_inlier = X[y == 0].values
        X_outlier = X[y == 1].values

        ss = ShuffleSplit(n_splits=1, train_size=val_ratio, random_state=SEED)
        (X_out_val_idx, X_out_test_idx) = next(ss.split(X_outlier))

        # TODO: Change test size to reflect original class balances
        # something like test_sz * original_ratio
        ss = ShuffleSplit(n_splits=1, test_size=len(X_out_test_idx), random_state=SEED)
        (X_in_train, X_in_test) = next(ss.split(X_inlier))

        out_val_sz = len(X_out_val_idx)
        X_train = X_inlier[X_in_train]
        X_in_test = X_inlier[X_in_test]

        # Create an additional validation split for training data
        X_in_val = X_train[:out_val_sz]
        X_train = X_train[out_val_sz:]

        X_out_val = X_outlier[X_out_val_idx]
        X_out_test = X_outlier[X_out_test_idx]

    logging.warn(
        f"# Samples: {len(X_train)} train, {len(X_out_val)} validation, {len(X_out_test)} test"
    )

    return X_train, (X_in_val, X_out_val), (X_in_test, X_out_test)
