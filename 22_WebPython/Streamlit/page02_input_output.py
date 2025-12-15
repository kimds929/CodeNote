import streamlit as st
import utils

def show():
    st.header("페이지 2")
    st.write("여기는 두 번째 페이지입니다.")
    utils.common_message()
    
    ########################################################################################
    # (Input Widget)
    i1 = st.text_input("문자 입력")

    sample = {"sample_question" : "What should I do with my girlfriend tomorrow?"}
    auto_complete = st.toggle("☘️어떻게 질문해야 할지 모르겠나요?   왼쪽 토글을 누르면 예시 질문과 답을 볼 수 있어요!☘️")
    with st.form(key="form"):
            text_input = st.text_input(
            label='"어디", "누구랑", "무엇을" 하고 싶은지 자세히 적어주시면 더 정확한 결과를 얻을 수 있어요!', 
            value = sample["sample_question"] if auto_complete else ""
            )
            submit_button = st.form_submit_button(label='Lucky Today!')

    if submit_button:
        if not text_input:
            st.error("질문을 입력해 주세요!")
        elif len(text_input) < 5:
            st.error("질문을 조금 더 자세하게 적어주세요!")
        else:
            st.success("오늘은 이런걸 해보는게 어떨까요? 🥳")


    i2 = st.number_input("숫자 입력", min_value=0, max_value=100)
    i3 = st.selectbox("선택", ["A", "B", "C"])
    i4 = st.checkbox("체크박스")
    page = st.radio("페이지 선택", ["홈", "분석", "설정"])
    st.write(f"선택한 페이지: {page}")

    i5 = st.button("버튼")
    st.divider()
    
    ########################################################################################
    # (Text Ouput)
    st.header("헤더")
    st.subheader("서브헤더")
    st.write("텍스트 출력")
    st.markdown("**마크다운** 지원")
    st.markdown('<a href="https://www.github.com/kimds929" target="_blank"><button>github_바로가기</button></a>', unsafe_allow_html=True)
    st.divider()
    
    
    
    # 1. 텍스트 입력
    #     st.text_input() : 한 줄 텍스트 입력
    #     st.text_area() : 여러 줄 텍스트 입력
    # 2. 숫자 입력
    #     st.number_input() : 정수 또는 실수 입력 가능, 최소·최대값과 step 설정 가능
    # 3. 버튼 및 액션
    #     st.button() : 클릭 시 특정 동작 실행
    #     st.download_button() : 파일 다운로드 제공
    # 4. 선택형 입력
    #     st.selectbox() : 드롭다운 형태의 단일 선택
    #     st.multiselect() : 다중 선택 가능
    #     st.radio() : 라디오 버튼 형태의 단일 선택
    #     st.checkbox() : 체크박스 형태의 True/False 입력
    # 5. 슬라이더
    #     st.slider() : 범위 내에서 숫자 선택 (단일 값 또는 범위)
    #     st.select_slider() : 지정된 옵션 중에서 슬라이드 선택
    # 6. 날짜·시간 입력
    #     st.date_input() : 날짜 선택
    #     st.time_input() : 시간 선택
    # 7. 파일 업로드
    #     st.file_uploader() : 로컬 파일 업로드 (CSV, 이미지 등)
    # 8. 색상 선택
    #     st.color_picker() : 색상 선택기 제공
    
    
        
    # # 1. 텍스트 입력
    # single_line_text = st.text_input("한 줄 텍스트 입력", "기본값")
    # multi_line_text = st.text_area("여러 줄 텍스트 입력", "여기에 입력하세요")

    # # 2. 숫자 입력
    # number_value = st.number_input("숫자 입력", min_value=0, max_value=100, step=1)

    # # 3. 버튼 및 액션
    # if st.button("버튼 클릭"):
    #     st.write("버튼이 클릭되었습니다!")

    # # 다운로드 버튼 예시
    # sample_data = "POSCO AI Assistant 예제 데이터"
    # st.download_button("데이터 다운로드", sample_data, file_name="sample.txt")

    # # 4. 선택형 입력
    # select_option = st.selectbox("드롭다운 선택", ["옵션 1", "옵션 2", "옵션 3"])
    # multi_select_option = st.multiselect("다중 선택", ["A", "B", "C"])
    # radio_option = st.radio("라디오 버튼 선택", ["Yes", "No"])
    # checkbox_value = st.checkbox("체크박스 선택")

    # # 5. 슬라이더
    # slider_value = st.slider("슬라이더 선택", min_value=0, max_value=100, value=50)
    # select_slider_value = st.select_slider("옵션 슬라이드 선택", options=["Low", "Medium", "High"])

    # # 6. 날짜·시간 입력
    # date_value = st.date_input("날짜 선택")
    # time_value = st.time_input("시간 선택")

    # # 7. 파일 업로드
    # uploaded_file = st.file_uploader("파일 업로드", type=["csv", "png", "jpg"])
    # if uploaded_file is not None:
    #     st.write("업로드된 파일 이름:", uploaded_file.name)

    # # 8. 색상 선택
    # color_value = st.color_picker("색상 선택", "#00f900")

    # # 출력 예시
    # st.write("입력된 한 줄 텍스트:", single_line_text)
    # st.write("입력된 여러 줄 텍스트:", multi_line_text)
    # st.write("선택된 숫자:", number_value)
    # st.write("드롭다운 선택:", select_option)
    # st.write("다중 선택:", multi_select_option)
    # st.write("라디오 선택:", radio_option)
    # st.write("체크박스 상태:", checkbox_value)
    # st.write("슬라이더 값:", slider_value)
    # st.write("옵션 슬라이더 값:", select_slider_value)
    # st.write("선택한 날짜:", date_value)
    # st.write("선택한 시간:", time_value)
    # st.write("선택한 색상:", color_value)