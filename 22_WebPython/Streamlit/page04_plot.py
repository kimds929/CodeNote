import streamlit as st
import utils
# pip install streamlit-drawable-canvas


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from utils_image import ImageRectAnnotator

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

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
    img_np = np.random.randint(0, 255, (200, 300))
    fig = plt.figure()
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")
    
    def on_submit(payload):
        # 여기에서 "지정된 곳으로 send" 처리
        # 예: st.session_state["rect_payload"] = payload
        # st.session_state["rect_payload"] = payload
        print(payload)
    
    options = [
        {"type": "radio", "key": "radio", "label": "", "choices": ["A", "B", "C"], "default": "A", "horizontal":True},
        # {"type": "selectbox", "key": "label", "label": "라벨", "choices": ["A", "B", "C"], "default": "A"},
        {"type": "checkbox", "key": "verified", "label": "검증됨", "default": False},
        # {"type": "multiselect", "key": "tags", "label": "태그", "choices": ["x", "y", "z"]},
        {"type": "text_input", "key": "memo", "label": "메모"},
    ]

    def on_submit(payload):
        st.session_state["submitted"] = payload
        print(payload)
        # {
        # "images": [
        #     {"id": 1, "file_name": "img001.jpg", "width": 1920, "height": 1080}
        # ],
        # "annotations": [
        #     {"id": 10, "image_id": 1, "bbox_xyxy": [100, 200, 400, 600], "label": "person"}
        # ]
        # }

    ann = ImageRectAnnotator(
        key_prefix="rect_demo",
        canvas_size=(500, 300),
        options=options,
        element=None,  # 비어도 OK (options만 있으면 UI 뜸)
        # element_kwargs={"label": "B", "memo": "초기 메모"},  # 일부 초기값 주입
        on_submit=on_submit,
    )

    ann.render(fig, orig_size=(300, 200), show_debug=False)

    if "submitted" in st.session_state:
        st.write("최종 payload:", st.session_state["submitted"])
    st.divider()
