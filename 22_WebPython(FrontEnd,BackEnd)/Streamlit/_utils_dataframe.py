import streamlit as st
import streamlit.components.v1 as components

import base64
import json
from datetime import datetime
import numpy as np
import pandas as pd

remote_library_url = 'https://raw.githubusercontent.com/kimds929'
try:
    import httpimport
    with httpimport.remote_repo(f"{remote_library_url}/CodeNote/blob/main/00_DataAnalysis_Basic/"):
        from DS_Basic_Module import DF_Summary, SummaryPlot, img_to_clipboard
except:
    import requests
    response = requests.get(f"{remote_library_url}/CodeNote/refs/heads/main/00_DataAnalysis_Basic/DS_Basic_Module.py", verify=False)
    exec(response.text)

from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, DataReturnMode


# 공통 Toast JS
toast_js = """
<script>
function showToast(message) {
    const toast = document.createElement("div");
    toast.textContent = message;
    toast.style.position = "fixed";
    toast.style.bottom = "20px";
    toast.style.right = "20px";
    toast.style.background = "rgba(0,0,0,0.85)";
    toast.style.color = "#fff";
    toast.style.padding = "10px 20px";
    toast.style.borderRadius = "5px";
    toast.style.fontSize = "14px";
    toast.style.zIndex = "9999";
    toast.style.boxShadow = "0 2px 6px rgba(0,0,0,0.3)";
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 2000);
}
</script>
"""

def st_to_clipboard_button(dataframe, button_text="📋", complete_text="Complete to clipboard!", height=40, index=False):
    # DataFrame을 문자열로 변환 (탭 구분)
    if index:
        df_csv = dataframe.to_csv(index=False, sep='\t')
    else:
        df_csv = dataframe.drop('index', axis=1).to_csv(index=False, sep='\t')
    df_json =json.dumps(df_csv)
    
    copy_js = f"""
        {toast_js}
        <script>
        const data = {df_json};
        function copyData(){{
            if (navigator.clipboard && window.isSecureContext){{
                navigator.clipboard.writeText(data).then(function(){{
                    showToast(`{complete_text}`);
                }});
            }} else {{
                const textarea = document.createElement("textarea");
                textarea.value = data;
                textarea.style.position = "fixed";
                textarea.style.left = "-9999px";
                document.body.appendChild(textarea);
                textarea.focus();
                textarea.select();
                document.execCommand("copy");
                document.body.removeChild(textarea);
                showToast(`{complete_text}`);
            }}
        }}
        </script>
        <style>
            .st-clipboradbtn {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-weight: 400;
                padding: 0.25rem 0.5rem;
                border-radius: 0.25rem;
                border: 1px solid rgba(49, 51, 63, 0.2);
                background-color: rgb(255, 255, 255);
                color: rgb(49, 51, 63);
                cursor: pointer;
                line-height: 1.2;
                font-size: 12px;
                height: 30px;
                margin-right: 4px;
                text-decoration: none;
            }}
            .st-clipboradbtn:hover {{
                background-color: rgb(230, 232, 236);
            }}
            body {{
                margin: 0;
            }}
        </style>
        <button onclick="copyData()" class="st-clipboradbtn">{button_text}</button>
    """
    # st.markdown(copy_js, unsafe_allow_html=True)
    components.html(copy_js, height=height)


def st_download_button(dataframe, button_text="📥", post_fix=None, height=40):
    # DataFrame → CSV (utf-8-sig 인코딩)
    csv_data = dataframe.to_csv(index=False, sep='\t').encode('utf-8-sig')
    b64 = base64.b64encode(csv_data).decode()
    if post_fix is None:
        filename = f"data.csv"
    else:
        filename = f"data_{post_fix}.csv"
    

    html_code = f"""
        <style>
            .st-btn {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-weight: 400;
                padding: 0.25rem 0.5rem;
                border-radius: 0.25rem;
                border: 1px solid rgba(49, 51, 63, 0.2);
                background-color: rgb(255, 255, 255);
                color: rgb(49, 51, 63);
                cursor: pointer;
                line-height: 1.2;
                font-size: 12px;
                height: 20px;
                margin-right: 4px;
                text-decoration: none;
            }}
            .st-btn:hover {{
                background-color: rgb(230, 232, 236);
            }}
            body {{
                margin: 0;
            }}
        </style>
        <a download="{filename}" href="data:text/csv;base64,{b64}" class="st-btn">{button_text}</a>
    """
    components.html(html_code, height=height)

def st_clipboard_download_button(dataframe, download_post_fix=None, complete_clipboard_text="Complete to clipboard!"):
    # CSV 데이터 준비
    df_csv = dataframe.drop('index', axis=1).to_csv(index=False, sep='\t')
    df_json = json.dumps(df_csv)
    csv_data = dataframe.to_csv(index=False, sep='\t').encode('utf-8-sig')
    b64 = base64.b64encode(csv_data).decode()
    filename = f"data.csv" if download_post_fix is None else f"data_{download_post_fix}.csv"

    html_code = f"""
        {toast_js}
        <script>
        const data = {df_json};
        function copyData(){{
            if (navigator.clipboard && window.isSecureContext){{
                navigator.clipboard.writeText(data).then(function(){{
                    showToast(`{complete_clipboard_text}`);
                }});
            }} else {{
                const textarea = document.createElement("textarea");
                textarea.value = data;
                textarea.style.position = "fixed";
                textarea.style.left = "-9999px";
                document.body.appendChild(textarea);
                textarea.focus();
                textarea.select();
                document.execCommand("copy");
                document.body.removeChild(textarea);
                showToast(`{complete_clipboard_text}`);
            }}
        }}
        </script>
        <style>
            .st-btn {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-weight: 400;
                padding: 0.25rem 0.5rem;
                border-radius: 0.25rem;
                border: 1px solid rgba(49, 51, 63, 0.2);
                background-color: rgb(255, 255, 255);
                color: rgb(49, 51, 63);
                cursor: pointer;
                line-height: 1.2;
                font-size: 14px;
                height: 32px;
                margin-right: 6px;
                text-decoration: none;
                box-sizing: border-box;
            }}
            .st-btn:hover {{
                background-color: rgb(230, 232, 236);
            }}
            /* a 태그와 button 태그의 기본 스타일 초기화 */
            .st-btn,
            .st-btn:link,
            .st-btn:visited {{
                text-decoration: none;
            }}

            .st-btn:focus {{
                outline: none;
            }}

            .st-btn-button,
            .st-btn-link {{
                all: unset;
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }}
        </style>
        <div class="btn-container">
            <button onclick="copyData()" class="st-btn">📋</button>
            <a download="{filename}" href="data:text/csv;base64,{b64}" class="st-btn">📥</a>
        </div>
    """
    components.html(html_code, height=50)


class AgGridTable:
    def __init__(
        self,
        aggrid_df=None,
        page_size=20,
        selection_mode='multiple',
        theme='streamlit',
        enable_EDA=True,
        enable_enterprise_modules=True,
        min_column_width=30,
        index_column=None,
        index_col_width=50,
        index_min_width=40,
        index_max_width=60,
        index_header='#',
        index_bg_color="#f8f9fa",
        index_font_weight="bold",
        index_text_align="center",
        text_filter_cols=None,
        height=650,
        **kwargs
    ):
        self.aggrid_df = aggrid_df.copy() if aggrid_df is not None else pd.DataFrame()
        self.df_columns = self.aggrid_df.columns
        self.page_size = page_size
        self.selection_mode = selection_mode
        self.theme = theme
        self.enable_EDA = enable_EDA
        self.enable_enterprise_modules = enable_enterprise_modules
        self.min_column_width = min_column_width
        self.index_col_width = index_col_width
        self.index_min_width = index_min_width
        self.index_max_width = index_max_width
        self.index_header = index_header
        self.index_bg_color = index_bg_color
        self.index_font_weight = index_font_weight
        self.index_text_align = index_text_align
        self.text_filter_cols = text_filter_cols or []
        self.height = height
        self.extra_options = kwargs

        self.index_column = index_column
        if not self.aggrid_df.empty:
            self._prepare_dataframe()

    def format_bytes(self, size):
        """
        바이트 단위의 숫자를 받아서
        Byte, KB, MB, GB, TB 단위로 자동 변환하여 문자열로 반환
        """
        # 단위 목록
        units = ['Bytes', 'KB', 'MB', 'GB', 'TB']
        index = 0
        
        # 1024로 나누면서 단위 변경
        while size >= 1024 and index < len(units) - 1:
            size /= 1024
            index += 1
        
        return f"{size:.2f} {units[index]}"

    # index 컬럼 추가
    def _prepare_dataframe(self):
        if 'index' not in self.aggrid_df.columns:
            self.aggrid_df.insert(0, 'index', self.aggrid_df.index)
            self.index_column = 'index'

    # update_dataframe
    def update_dataframe(self, df: pd.DataFrame):
        """외부에서 DataFrame을 업데이트"""
        self.aggrid_df = df.copy()
        self.df_columns = self.aggrid_df.columns
        self._prepare_dataframe()
    
    # get column filters
    def _get_column_filters(self, default_columns=None):
        if default_columns is None:
            default_columns = list(self.df_columns)
        
        # 멀티셀렉트의 key 지정
        columns_filter = st.multiselect(
            "Columns",
            options=self.df_columns,
            default=default_columns,
            key="aggrid_columns_filter"
        )
        columns_filter = [self.index_column] + columns_filter
        return columns_filter

    # build grid options
    def _build_grid_options(
        self,
        dataframe,
        index_sortable=True,
        index_col_width=None,
        index_min_width=None,
        index_max_width=None,
        index_header=None,
        index_bg_color=None,
        index_font_weight=None,
        index_text_align=None,
        min_column_width=None,
        page_size=None,
        selection_mode=None,
        text_filter_cols=None
        ):
        # None일 경우 __init__에서 지정된 값 사용
        index_col_width = self.index_col_width if index_col_width is None else index_col_width
        index_min_width = self.index_min_width if index_min_width is None else index_min_width
        index_max_width = self.index_max_width if index_max_width is None else index_max_width
        index_header = self.index_header if index_header is None else index_header
        index_bg_color = self.index_bg_color if index_bg_color is None else index_bg_color
        index_font_weight = self.index_font_weight if index_font_weight is None else index_font_weight
        index_text_align = self.index_text_align if index_text_align is None else index_text_align
        min_column_width = self.min_column_width if min_column_width is None else min_column_width
        page_size = self.page_size if page_size is None else page_size
        selection_mode = self.selection_mode if selection_mode is None else selection_mode
        text_filter_cols = self.text_filter_cols if text_filter_cols is None else text_filter_cols

        gb = GridOptionsBuilder.from_dataframe(dataframe)

        # index 컬럼 설정
        gb.configure_column(
            self.index_column,
            header_name=index_header,
            filter=True,
            sortable=index_sortable,
            editable=False,
            width=index_col_width,
            min_width=index_min_width,
            max_width=index_max_width,
            pinned='left',
            cellStyle={
                "backgroundColor": index_bg_color,
                "fontWeight": index_font_weight,
                "padding": "0px 2px",
                "textAlign": index_text_align
            }
        )

        # 기본 컬럼 설정
        gb.configure_default_column(
            editable=False,
            enablePivot=True,
            enableRowGroup=True,
            enableValue=True,
            filterable=True,
            groupable=True,
            sortable=False,
            filter='agSetColumnFilter',
            enable_filtering=True,
            wrapText=True,
            minWidth=min_column_width,
            flex=1,
            resizable=True,
            suppressMenu=False
        )

        # 특정 컬럼에 텍스트 필터 적용
        for col in text_filter_cols:
            if col in self.df_columns:
                gb.configure_column(col, filter="agTextColumnFilter")

        # 선택 기능
        gb.configure_selection(
            selection_mode=selection_mode,
            suppressRowDeselection=False
        )
            
        # 페이지네이션
        gb.configure_pagination(
            enabled=True,
            paginationAutoPageSize=False,
            paginationPageSize=page_size
        )

        # Grid 옵션
        gb.configure_grid_options(
            domLayout='normal',
            pivotMode=False,
            cellSelection=True,
            rowSelection=selection_mode,
            enableRangeSelection=True,
            pagination=True,
            paginationAutoPageSize=False
        )

        # 사이드바
        gb.configure_side_bar(
            filters_panel=True,
            columns_panel=True
        )

        return gb

    def _EDA_options(self, dataframe):
        with st.expander("EDA Options"):
            st_columns_1, st_columns_2 = st.columns([3,1])
            
            df_sumamry =  DF_Summary(dataframe.drop('index', axis=1, errors='ignore'), n_samples=40)
            with st_columns_1:
                summary_table = pd.DataFrame(df_sumamry.summary).copy()
                for col in summary_table.select_dtypes(include=['object']).columns:
                    summary_table[col] = summary_table[col].astype(str)
                summary_table = summary_table.reset_index()
                
                st_columns_1_1, st_columns_1_2 = st.columns([9,1])
                with st_columns_1_2:
                    st_to_clipboard_button(summary_table, index=True)
                summary_gb = self._build_grid_options(summary_table, index_sortable=False, page_size=9999)
                summary_gb.configure_column('dtype', editable=True, cellEditor='agSelectCellEditor',
                                            cellEditorParams={
                                                'values':['object', 'int', 'float', 'bool', 'datetime']
                                            })
                SummaryGridOptins = summary_gb.build()
                summary_grid_response = AgGrid(summary_table
                                            ,gridOptions = SummaryGridOptins
                                            )
            
            selected_columns = None
            with st_columns_2:
                if summary_grid_response.selected_data is not None:
                    selected_columns = list(summary_grid_response.selected_data['index'])
                    fig = df_sumamry.summary_plot(on=selected_columns, return_plot=True)
                    
                    st_columns_2_1, st_columns_2_2 = st.columns([8,2])
                    with st_columns_2_2:
                        if st.button("📋"):
                            img_to_clipboard(fig) 
                            st.toast("Complete img to clipboard!")
                    st.pyplot(fig)
    
    def render(self, dataframe=None, **kwargs):
        """df 인자를 주면 그걸 사용, 없으면 기존 저장된 df 사용"""
        if dataframe is not None:
            self.update_dataframe(dataframe)

        # [Header] -------------------------------------------------------------------
        st_columns_1, st_columns_2, st_columns_3, st_columns_4 = st.columns([7, 0.5, 1, 1.5])

        reset_clicked = False
        with st_columns_2:
            st.markdown('<p></p>', unsafe_allow_html=True)
            if st.button("🔄"):
                st.session_state["aggrid_columns_filter"] = list(self.df_columns)
                reset_clicked = True

        with st_columns_1:
            if reset_clicked:
                columns_filter = self._get_column_filters(default_columns=list(self.df_columns))
            else:
                columns_filter = self._get_column_filters()

        # ✅ 필터링 로직 수정: 선택된 컬럼이 없으면 원본 전체 표시
        if len(columns_filter) > 1:  # index_column 외에 선택된 컬럼이 있는 경우
            df_filtered = self.aggrid_df[columns_filter]
        else:
            df_filtered = self.aggrid_df.copy()

        

        # [EDA] -------------------------------------------------------------------
        if self.enable_EDA:
            self._EDA_options(df_filtered)

        # [Main] -------------------------------------------------------------------
        columns_sorted = [self.index_column] + \
                [col for col in columns_filter if col != self.index_column] + \
                [x for x in self.df_columns if x not in columns_filter]

        
        grid_gb = self._build_grid_options(self.aggrid_df[columns_sorted])
        gridOptions = grid_gb.build()
        grid_response = AgGrid(
            df_filtered,
            gridOptions=gridOptions,
            # data_return_mode='AS_INPUT',
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            # columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
            # columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
            update_mode='MODEL_CHANGED',
            fit_columns_on_grid_load=False,
            enable_enterprise_modules=self.enable_enterprise_modules,
            height=self.height,
            theme=self.theme,
            
            **{**self.extra_options, **kwargs}
        )
        
        # [Header : for filter] -------------------------------------------------------------------
        df_after_select_filter = grid_response['data']
        # if grid_response.rows_id_after_filter is not None:
        #     # df_after_select_filter = df_filtered.loc[np.array(grid_response.rows_id_after_filter).astype(int)]
        #     df_after_select_filter = grid_response['data']
        # else:
        #     df_after_select_filter = df_filtered.copy()
        
        with st_columns_3:
            now_date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.write(" ")
            st_clipboard_download_button(df_after_select_filter, download_post_fix=now_date_str)

        with st_columns_4:
            df_memory = self.format_bytes(df_after_select_filter.memory_usage().sum())
            df_shape = df_after_select_filter.shape
            st.markdown(
                f"<p>· memory : {df_memory}<br>· shape : {df_shape}</p>",
                unsafe_allow_html=True
            )
            
        return grid_response
    



# df_columns = aggrid_df.columns
# aggrid_df.insert(0, 'index', aggrid_df.index)
# columns_filter = st.multiselect("Column_Selection", options=df_columns, default=df_columns)
# columns_filter = ['index'] + columns_filter
# # with st.expander("Settings"):
# columns_sorted = ['index'] + [x for x in df_columns if x in columns_filter] + [x for x in df_columns if x not in columns_filter]

# # GridOptionsBuilder를 사용하여 세부적인 Ag-Grid 옵션을 설정합니다.
# gb = GridOptionsBuilder.from_dataframe(aggrid_df[columns_sorted])

# # 필터링 기능을 활성화합니다. 각 열에 드롭다운 필터 메뉴가 생깁니다.
# gb.configure_column('index'
#                     ,header_name='#'
#                     ,filter=True
#                     ,sortable=True
#                     ,editable=False
#                     ,width=50
#                     ,min_width=40
#                     ,max_width=60
#                     ,pinned='left'
#                     ,cellStyle={
#                         "backgroundColor": "#f8f9fa"
#                         ,"fontWeight": "bold"
#                         ,"padding": "0px 2px"                            
#                         ,"textAlign":"center"
#                     }
#                     )

# gb.configure_default_column(
#     editable=False     # cell수정가능
#     ,enablePivot=True  # 피벗 가능
#     ,enableRowGroup=True    # 피벗 Row group
#     ,enableValue=True   # 피벗 Value기능
#     ,filterable=True    # filter기능
#     ,groupable=True 
#     # ,enable_ordering=True
#     ,sortable=False
#     ,filter='agSetColumnFilter'
#     ,enable_filtering=True
#     ,wrapText=True
#     # ,autoHegiht=True
#     # ,autoWidth=True
#     ,min_column_width=100   # 너무 좁아지지 않도록 하한설정
#     ,flex = 0      # 모든 column 폭을 grid 폭 기준으로 비율나눔
#     ,resizable=True
#     ,suppressMenu=False      # 메뉴 숨김 방지
# )

# # # 명시적으로 문자열 열에 'agTextColumnFilter' 사용을 지시합니다.
# # # 이렇게 하면 사이드바의 'Filters' 탭에서 해당 열에 대한 검색창이 활성화됩니다.
# # for c_cat in ['pclass','survived','sex','sibsp','parch','embarked']:
# #     gb.configure_column(c_cat, filter="agTextColumnFilter")

# # selection 설정
# gb.configure_selection(
#     selection_mode='multiple'   # 'single | 'multiple' | 'disable'
#     # ,use_checkbox=True
#     # ,rowMultiSelectWithClick=True
#     ,suppressRowDeselection=False
# )

# # pagination
# gb.configure_pagination(
#     enabled=True
#     ,paginationAutoPageSize=False
#     ,paginationPageSize=20
# )

# # grid option
# gb.configure_grid_options(domLayout='normal' 
#                         ,pivotMode=False  # Pivot Mode 켜기
#                         ,cellSelection=True
#                         ,rowSelection='multiple'  # 행 선택 가능
#                         ,enableRangeSelection=True  # 셀 범위 선택 가능
#                         ,pagination=True
#                         ,paginationAutoPageSize=False)
# gridOptions = gb.build()

# # 사이드바에 필터 창을 표시하도록 설정
# gb.configure_side_bar(
#     filters_panel=True
#     ,columns_panel=True
# ) 

# # AgGrid 컴포넌트 렌더링
# # key='grid1'을 사용하여 여러 AgGrid 인스턴스를 구분할 수 있습니다.
# st.write(df2.shape)
# grid_response = AgGrid(
#     aggrid_df[columns_filter]
#     ,gridOptions=gridOptions
#     ,data_return_mode='AS_INPUT'
#     ,columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS
#     ,update_mode='MODEL_CHANGED'    # 필터 변경 시 streamlit으로 상태를 반환
#     ,fit_columns_on_grid_load=True
#     ,enable_enterprise_modules=True
#     ,height=650
#     # ,width='100%'
#     ,theme='streamlit' # 'streamlit','alpine', 'balham', 'material'
# )
    
# # st.write(grid_response.selected_rows)
# # st.write(grid_response.grid_state['focusedCell'])
# # st.write(grid_response.grid_state)



# # 0:"_AgGridReturn__component_value_set"
# # 1:"_AgGridReturn__conversion_errors"
# # 2:"_AgGridReturn__data_return_mode"
# # 3:"_AgGridReturn__get_data"
# # 4:"_AgGridReturn__get_dataGroups"
# # 5:"_AgGridReturn__original_data"
# # 6:"_AgGridReturn__process_grouped_response"
# # 7:"_AgGridReturn__process_vanilla_df_response"
# # 8:"_AgGridReturn__try_to_convert_back_to_original_types"
# # 9:"__abstractmethods__"
# # 10:"__class__"
# # 11:"__class_getitem__"
# # 12:"__contains__"
# # 13:"__delattr__"
# # 14:"__dict__"
# # 15:"__dir__"
# # 16:"__doc__"
# # 17:"__eq__"
# # 18:"__format__"
# # 19:"__ge__"
# # 20:"__getattribute__"
# # 21:"__getitem__"
# # 22:"__gt__"
# # 23:"__hash__"
# # 24:"__init__"
# # 25:"__init_subclass__"
# # 26:"__iter__"
# # 27:"__le__"
# # 28:"__len__"
# # 29:"__lt__"
# # 30:"__module__"
# # 31:"__ne__"
# # 32:"__new__"
# # 33:"__orig_bases__"
# # 34:"__parameters__"
# # 35:"__reduce__"
# # 36:"__reduce_ex__"
# # 37:"__repr__"
# # 38:"__reversed__"
# # 39:"__setattr__"
# # 40:"__sizeof__"
# # 41:"__slots__"
# # 42:"__str__"
# # 43:"__subclasshook__"
# # 44:"__weakref__"
# # 45:"_abc_impl"
# # 46:"_is_protocol"
# # 47:"_set_component_value"
# # 48:"columns_state"
# # 49:"data"
# # 50:"dataGroups"
# # 51:"event_data"
# # 52:"get"
# # 53:"grid_options"
# # 54:"grid_response"
# # 55:"grid_state"
# # 56:"items"
# # 57:"keys"
# # 58:"rows_id_after_filter"
# # 59:"rows_id_after_sort_and_filter"
# # 60:"selected_data"
# # 61:"selected_dataGroups"
# # 62:"selected_rows"
# # 63:"selected_rows_id"
# # 64:"values"
    