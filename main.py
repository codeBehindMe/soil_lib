from typing import Optional
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import numpy as np
from scipy.spatial import distance

from src.utils import filter_wavelength_columns, load_model
from src.constants import K_MEANS_MODEL_PATH, PATH_TO_TRAINING_SET, PATH_TO_TEST_SET


def plot_spectrometry(df: pd.DataFrame):

    df = df[["PIDN", *filter(lambda x: x.startswith("m"), df.columns.values)]]

    df_melt = df.melt(id_vars=["PIDN"])

    return px.line(
        df_melt,
        x="variable",
        y="value",
        color="PIDN",
        labels={"variable": "wavelength", "value": "reflectance"},
    )


def create_library(train_df: pd.DataFrame, model: KMeans) -> pd.DataFrame:
    predict_df = filter_wavelength_columns(train_df)

    return_df = train_df.copy(deep=True)
    return_df["CLASS"] = model.predict(predict_df)

    return return_df


def predict_class(df: pd.DataFrame, model: KMeans) -> int:
    x = filter_wavelength_columns(df)
    return int(model.predict(x)[0])


def get_k_closest_from_lib(
    selected_df: pd.DataFrame, lib_df: pd.DataFrame, model: KMeans, k: Optional[int] = 5
) -> pd.DataFrame:
    """
    Get the closest samples from the library
    """

    pred_class = predict_class(selected_df, model)

    class_df = lib_df[lib_df["CLASS"] == pred_class]

    target_vector = filter_wavelength_columns(selected_df).to_numpy().flatten()

    class_df["SCORES"] = filter_wavelength_columns(class_df).apply(
        lambda x: distance.euclidean(x.to_numpy().flatten(), target_vector), axis=1
    )

    return class_df.sort_values("SCORES", ascending=False).head(k)


if __name__ == "__main__":
    train_df = pd.read_csv(PATH_TO_TRAINING_SET)
    test_df = pd.read_csv(PATH_TO_TEST_SET)

    mdl = load_model(K_MEANS_MODEL_PATH)
    soil_lib = create_library(train_df, mdl)

    st.set_page_config(layout="wide")

    st.title("Spectroscopy library")

    col1, col2 = st.columns(2)

    col1.header("Your Data")
    col1.info("We're pretending that you've already uploaded your spectrometry data")

    selected_sample = col1.selectbox("Select Sample ID", test_df["PIDN"].values)
    selected_df = test_df[test_df["PIDN"] == selected_sample]

    col1.write(selected_df)
    col1.plotly_chart(plot_spectrometry(selected_df), use_container_width=True)

    k_closest = get_k_closest_from_lib(selected_df, soil_lib, mdl)

    col2.header("Library Matches")
    col2.write(k_closest)
    col2.plotly_chart(plot_spectrometry(k_closest), use_container_width=True)
