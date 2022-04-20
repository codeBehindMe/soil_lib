from typing import List
import pandas as pd
import numpy as np


def bucketize_wavelenghts_single_decimal_place(
    df: pd.DataFrame, id_col: str, wavelength_cols: List[str]
):
    """
    Bucketizes wavelenghts to single decimal places
    """

    df_subset = df[[id_col, *wavelength_cols]]
    df_melt = df_subset.melt(id_vars=[id_col])

    df_melt["wl"] = df.melt["variable"].apply(lambda x: float(x[1:]))

    df_melt["wl_round"] = np.round(df_melt["wl"])

    df_agg = (
        df_melt[[id_col, "wl_round", "value"]].groupby([id_col, "wl_round"]).agg("mean")
    )

    df_agg["wl_str"] = df_melt["wl_round"].apply(lambda x: f"m{x}")

    raise NotImplementedError()
