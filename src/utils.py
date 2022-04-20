from typing import Optional
from joblib import dump, load
import pandas as pd
from sklearn.cluster import KMeans


def save_model(model: KMeans, file_name: str):

    dump(model, file_name)
    return


def load_model(file_name: str) -> KMeans:
    return load(file_name)


def filter_wavelength_columns(
    df: pd.DataFrame, wl_col_id: Optional[str] = "m"
) -> pd.DataFrame:
    return df[filter(lambda x: x.startswith(wl_col_id), df.columns.values)]
