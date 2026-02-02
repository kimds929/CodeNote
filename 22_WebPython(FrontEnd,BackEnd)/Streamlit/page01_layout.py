import streamlit as st
import utils  # utils.py의 함수 사용 가능


def show():
    st.header("페이지 1")
    st.write("여기는 첫 번째 페이지입니다.")
    utils.common_message()
    
    
    ########################################################################################
    # (영역나누기)

    # Column Basic
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("왼쪽 영역")
        st.write("여기에 내용 작성")

    with col2:
        st.subheader("오른쪽 영역")
        st.write("여기에 내용 작성")
    st.divider()

    # Column ratio
    col1, col2, col3 = st.columns([1, 2, 1])  # 비율 지정 가능
    with col1:
        st.write("왼쪽")
    with col2:
        st.write("가운데")
    with col3:
        st.write("오른쪽")
    st.divider()


    # Tab
    tab1, tab2 = st.tabs(["📊 데이터", "⚙ 설정"])
    with tab1:
        st.write("데이터 페이지")
    with tab2:
        st.write("설정 페이지")

    with st.expander("자세히 보기"):
        st.write("이 내용은 클릭 시 펼쳐집니다.")
    st.divider()


    # Container
    container = st.container()
    container.write("이건 컨테이너 안에 있는 내용")
    st.write("이건 컨테이너 밖의 내용")

    ########################################################################################
    # (Basic)
    # -----------------------------------------------------------------------------------
    name = st.text_input("이름을 입력하세요")
    if st.button("인사하기"):
        st.write(f"안녕하세요, {name}님!")
    st.divider()
    # -----------------------------------------------------------------------------------