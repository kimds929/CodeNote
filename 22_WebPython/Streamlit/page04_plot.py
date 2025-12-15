import streamlit as st
import utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode


def show():
    ########################################################################################
    # (Pyplot)
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig = plt.figure(figsize=(5,3))
    plt.plot(x, y)
    plt.close()
    st.pyplot(fig)
    st.divider()

    # 샘플 데이터
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 30, 20, 15],
        "category": ["A", "B", "A", "B", "A"]
    })

    # Plotly 그래프 생성
    fig = px.scatter(df, x="x", y="y", color="category", title="인터랙티브 Scatter Plot")

    # Streamlit에 표시
    st.plotly_chart(fig, use_container_width=True)
    st.divider()
