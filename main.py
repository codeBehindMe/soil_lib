import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

from src.utils import filter_wavelength_columns, load_model
from src.constants import K_MEANS_MODEL_PATH, PATH_TO_TRAINING_SET, PATH_TO_TEST_SET


def plot_spectrometry(df: pd.DataFrame):

    df = df[["PIDN", *filter(lambda x: x.startswith("m"), df.columns.values)]]

    df_melt = df.melt(id_vars=["PIDN"])

    return px.line(df_melt, x="variable", y="value", color="PIDN")


def create_library(train_df: pd.DataFrame, model: KMeans) -> pd.DataFrame:
    predict_df = filter_wavelength_columns(train_df)

    return_df = train_df.copy(deep=True)
    return_df["CLASS"] = model.predict(predict_df)

    return return_df


if __name__ == "__main__":
    train_df = pd.read_csv(PATH_TO_TRAINING_SET)
    test_df = pd.read_csv(PATH_TO_TEST_SET)

    mdl = load_model(K_MEANS_MODEL_PATH)
    soil_lib = create_library(train_df, mdl)

    st.title("Spectroscopy library")


    st.write(soil_lib)
    st.selectbox("Select Sample", train_df["PIDN"].values)
    # st.plotly_chart(plot_spectrometry(df))
