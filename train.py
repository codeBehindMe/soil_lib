import pandas as pd
from sklearn.cluster import KMeans
from src.utils import save_model, filter_wavelength_columns
from src.constants import (
    PATH_TO_TRAINING_SET,
    WAVELENGTH_COL_IDENTIFIER,
    K_MEANS_MODEL_PATH,
)

# Trains a K Means model to cluster the data into groups


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_model(train_pack: pd.DataFrame, **k_means_args) -> KMeans:

    mdl = KMeans(**k_means_args)
    return mdl.fit(train_pack)


if __name__ == "__main__":
    train_df = load_data(PATH_TO_TRAINING_SET)
    train_x = filter_wavelength_columns(train_df, WAVELENGTH_COL_IDENTIFIER)

    mdl = train_model(train_x)

    save_model(mdl, K_MEANS_MODEL_PATH)
