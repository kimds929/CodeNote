import streamlit as st
import utils
# pip install streamlit-drawable-canvas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



import plotly.express as px
import plotly.graph_objects as go

from utils_image import ImageRectAnnotator

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode



import io
import streamlit.components.v1 as components
import base64
from io import BytesIO
from PIL import Image




def send_rect(payload: dict):
    """
    '다른곳으로 send'를 여기서 처리.
    예: requests.post(url, json=payload)
    여기서는 데모로 session_state에 저장만 함.
    """
    pass
    # st.session_state["sent_payload"] = payload
    
def rect_signature(obj, nd=2):
    # fabric rect에서 보통 left/top/width/height 쓰임
    left = round(float(obj.get("left", 0.0)), nd)
    top = round(float(obj.get("top", 0.0)), nd)
    w = round(float(obj.get("width", 0.0)), nd)
    h = round(float(obj.get("height", 0.0)), nd)
    return (left, top, w, h)

def extract_rect_xyxy(obj):
    """
    Fabric rect object -> (x1,y1,x2,y2) in canvas pixels
    scale까지 반영해서 정확히 뽑기
    """
    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    w = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
    h = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))
    x1, y1 = left, top
    x2, y2 = left + w, top + h
    return x1, y1, x2, y2

def show():  
    
    def on_submit(payload):
        print(payload)
        st.session_state["submitted"] = payload
        st.write("✅ on_submit 호출됨")
        # st.json(payload)
    
    
    img_np = np.random.randint(0, 255, (200, 300))
    # img_np = np.arange(200*300).reshape(200,300)/(200*300)
    fig = plt.figure()
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")
    
    options = [
        # {"type": "radio", "key": "radio", "label": "", "choices": ["A", "B", "C"], "default": "A", "horizontal":True},
        {"type": "selectbox", "key": "label", "label": "라벨", "choices": ["A", "B", "C"], "default": "A"},
        {"type": "checkbox", "key": "verified", "label": "검증됨", "default": False},
        {"type": "multiselect", "key": "tags", "label": "태그", "choices": ["x", "y", "z"]},
        {"type": "text_input", "key": "memo", "label": "메모"},
    ]
    if st.button('rerun'):
        st.rerun()
    
    if "initialized" not in st.session_state:
        st.session_state.ann = ImageRectAnnotator(
            key_prefix="rect_demo",
            canvas_size=(500, 300),
            options=options,
            on_submit=on_submit,
        )
        st.session_state.initialized = True
    
    st.session_state.ann.add_payload({"study_id": "S123", "operator": "kimds929"})
    st.session_state.ann.render(fig, orig_size=(300, 200), 
               init_xyxy = [0, 50, 100, 100], 
            # show_debug=True
            )
    
    if st.button('set_box'):
        st.session_state.ann._reset()
        st.rerun()

        
    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    st.write(st.session_state)
    # if "submitted" in st.session_state:
    #     st.write("DEBUG coords:", st.session_state.get("rect_demo__coords"))
    #     st.write("DEBUG pending:", st.session_state.get("rect_demo__pending_submit"))
    st.divider()
