import pandas as pd
from sklearn.cluster import KMeans
from src.utils import save_model, filter_wavelength_columns

# Trains a K Means model to cluster the data into groups


PATH_TO_TRAINING_SET = "training.csv"
ID_COL = "PIDN"
WAVELENGTH_COL_IDENTIFIER = "m"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def create_training_package(
    df: pd.DataFrame, id_col: str, wl_col_identifier: str
) -> pd.DataFrame:
    """
    Use only the spectrometry for clustering.

    Removes other unrequired columns.
    """
    wl_cols = filter(lambda x: x.startswith(wl_col_identifier), df.columns.values)

    return df[wl_cols]


def train_model(train_pack: pd.DataFrame, **k_means_args) -> KMeans:

    mdl = KMeans(**k_means_args)
    return mdl.fit(train_pack)


if __name__ == "__main__":
    train_df = load_data(PATH_TO_TRAINING_SET)
    train_x = filter_wavelength_columns(train_df, WAVELENGTH_COL_IDENTIFIER)

    mdl = train_model(train_x)

    save_model(mdl, "k_means.mdl")
