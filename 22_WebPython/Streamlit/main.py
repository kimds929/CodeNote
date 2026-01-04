
import os
import streamlit as st
# "D:/DataScience/★GitHub_kimds929/CodeNote/22_WebPython/Streamlit/main.py"
# "D:/DataScience/★GitHub_kimds929/CodeNote/22_WebPython/Streamlit/"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

import psutil
import page01_layout
import page02_input_output
import page03_dataframe
import page04_plot

# import page05_query
 
# streamlit run app.py
# streamlit run test01.py --server.address=0.0.0.0 --server.port 8501

########################################################################################
# (Page Layout)
st.set_page_config(
    page_title="Streamlit Dashboard",   # 브라우저 탭 제목
    page_icon="📊",                 # 브라우저 탭 아이콘
    layout="wide",                  # 'centered' 또는 'wide'
    initial_sidebar_state="expanded" # 'auto', 'expanded', 'collapsed'
)

########################################################################################


########################################################################################
# (SideBar)
st.sidebar.title("메뉴")
# (Exit Button)
exit_app=st.sidebar.button("Close APP")
if exit_app:
    pid=os.getpid()
    p=psutil.Process(pid)
    p.terminate()


# 세션 상태 초기화
if "counter" not in st.session_state:
    st.session_state.counter = 0

initial_page = "page4"
# 기본페이지 설정
if "current_page" not in st.session_state:
    st.session_state.current_page = initial_page

# URL 파라미터 읽기
query_params = st.query_params
current_page = query_params.get("page", initial_page)


# st.markdown("""
#     <style>
#     div[data-testid="stSidebarContent"] div.stButton > button[kind="secondary"]:first-child {
#         background-color: #4CAF50;
#         color: white;
#         padding: 12px 24px;
#         font-size: 16px;
#         border-radius: 8px;
#         border: none;
#     }
#     div.stButton > button:hover {
#         background-color: #45a049;
#     }
#     </style>
# """, unsafe_allow_html=True)

st.sidebar.write("메뉴")
with st.container():
    st.markdown('<div class="main-menu-container">', unsafe_allow_html=True)
    if st.sidebar.button("메인"):
        st.session_state.current_page = "main"
    if st.sidebar.button("페이지 1 : Layout"):
        st.session_state.current_page = "page1"
    if st.sidebar.button("페이지 2 : Input/Output"):
        st.session_state.current_page = "page2"
    if st.sidebar.button("페이지 3 : DataFrame"):
        st.session_state.current_page = "page3"
    if st.sidebar.button("페이지 4 : Plot"):
        st.session_state.current_page = "page4"
    if st.sidebar.button("페이지 5 : Query"):
        st.session_state.current_page = "page5"
    st.markdown('</div>', unsafe_allow_html=True)

# 페이지별 내용
if st.session_state.current_page == "main":
    st.header("메인 페이지")
    st.write("여기는 메인 페이지입니다.")
    if st.button("카운터 증가"):
        st.session_state.counter += 1
    st.write(f"현재 카운터 값: {st.session_state.counter}")

elif st.session_state.current_page == "page1":
    page01_layout.show()

elif st.session_state.current_page == "page2":
    page02_input_output.show()

elif st.session_state.current_page == "page3":
    page03_dataframe.show()
    
elif st.session_state.current_page == "page4":
    page04_plot.show()



    
    


    