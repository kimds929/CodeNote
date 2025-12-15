import streamlit as st
import utils
from utils_dataframe import AgGridTable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px



# import os
# import sys
# sys.path.append(r'D:\DataScience\00_DataAnalysis_Basic')
# from DS_Basic_Module import DF_Summary

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode

def show():


########################################################################################
    # (DataFrame)

    # --------------------------------------------------------------------------------------

    # # pip install streamlit-aggrid
    # # 샘플 데이터
    data_url = 'https://raw.githubusercontent.com/kimds929/CodeNote/refs/heads/main/99_DataSet/Data_Tabular/'
    df = pd.read_csv(f'{data_url}/titanic.csv', encoding='utf-8-sig')
    # df_summary = DF_Summary(df)
    # df_summary.summary
    
    for c_cat in ['pclass','survived','sex','sibsp','parch','embarked']:
        df[c_cat] = df[c_cat].astype(str)
    # st.dataframe(df, use_container_width=True)
    # # AgGrid(df) 
    # st.divider()
    # ------------------------------------------------------------------------------------
    st.write("### Ag-Grid (사이드바 필터 활성화)")
    
    
    # 1) 세션에 초기 df가 없으면 한 번만 저장
    init_df = pd.DataFrame({'empty': [np.nan]})
    
    if "original_df" not in st.session_state:
        st.session_state["original_df"] = init_df.copy()

    # 2) 항상 session_df_key를 AgGridTable에 넘겨줌
    aggrid = AgGridTable(
        aggrid_df=st.session_state["original_df"],
        table_id="main_table",
        session_df_key="original_df",   # 🔥 중요
    )
    
    # 3) 클립보드 버튼
    if st.button("read_clipboard"):
        df_clip = pd.read_clipboard(sep='\t')
        st.session_state["original_df"] = df_clip.copy()
        # render에 바로 넘겨줘도 되고, 안 넘겨줘도 됨 (update_dataframe이 알아서 세션에 반영)
        response = aggrid.render(df_clip)
    else:
        response = aggrid.render(st.session_state["original_df"])
    
    # aggrid = AgGridTable(df)
    
    # if st.button("read_clipboard"):
    #     df = pd.read_clipboard(sep='\t')
    #     st.session_state['original_df'] = df  # 세션에 저장
    #     response = aggrid.render(df)
    # else:
    #     if 'original_df' in st.session_state:
    #         response = aggrid.render(st.session_state['original_df'])
    #     else:
    #         empty_dataframe = pd.DataFrame({'empty': [np.nan]})
    #         response = aggrid.render(empty_dataframe)
        


    