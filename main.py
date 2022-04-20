import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

df = pd.read_csv("training.csv")

df = df.head(5)


def plot_spectrometry(df: pd.DataFrame):

    df = df[["PIDN", *filter(lambda x: x.startswith("m"), df.columns.values)]]

    df_melt = df.melt(id_vars=["PIDN"])

    return px.line(df_melt, x="variable", y="value", color="PIDN")


if __name__ == "__main__":
    st.title("Spectroscopy library")
    
    st.selectbox("Select Sample", df["PIDN"].values)
    st.plotly_chart(plot_spectrometry(df))
