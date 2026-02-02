import sys
sys.path.append('D:/DataScience/★GitHub_kimds929/DS_Library')
sys.path.append('D:/DataScience/★GitHub_kimds929/CodeNote/00_DataAnalysis_Basic')

import streamlit as st
import streamlit.components.v1 as components

import base64
import json
from datetime import datetime
import numpy as np
import pandas as pd

from DS_Basic_Module import DF_Summary, SummaryPlot, img_to_clipboard

from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, DataReturnMode




############################################################################################################################################
# 허용가능한 max length를 추출해주는 함수
def alloable_len(dataframe, allowable_memory=2**10 * 2**10 * 200):
    lower_len = 0
    upper_len = allowable_max_len = dataframe.shape[0]
    if dataframe.memory_usage().sum().item() <= allowable_memory: 
        return len(dataframe)
    else:
        while True:
            memory = dataframe.iloc[:allowable_max_len].memory_usage().sum().item()
            # print(allowable_memory, memory, allowable_max_len)
            if memory > allowable_memory:
                # print('exceed')
                upper_len = allowable_max_len
                allowable_max_len = allowable_max_len // 2 
                
            elif memory < allowable_memory * 0.9:
                # print('below')
                lower_len = allowable_max_len
                allowable_max_len = (allowable_max_len + upper_len) // 2
            else:
                break
        return allowable_max_len
    

# byte를 단위환산
def format_bytes(size):
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

############################################################################################################################################


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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
def st_download_button_bigdata(dataframe, drop_index=True, button_text="📥", post_fix=None):
    if drop_index:
        csv_data = dataframe.drop('index', axis=1).to_csv(index=False, sep='\t').encode('utf-8-sig')
    else:
        csv_data = dataframe.to_csv(index=False, sep='\t').encode('utf-8-sig')
    filename = f"data_{post_fix}.csv" if post_fix else "data.csv"
    st.download_button(
        label=button_text,
        data=csv_data,
        file_name=filename,
        mime='text/csv'
    )



# ------------------------------------------------------------------------------------------------------------------------------------------------------------



def st_clipboard_download_button(dataframe, download_post_fix=None, drop_index=True, complete_clipboard_text="Complete to clipboard!"):
    # CSV 데이터 준비
    if drop_index:
        df_csv = dataframe.drop('index', axis=1).to_csv(index=False, sep='\t')
    else:
        df_csv = dataframe.to_csv(index=False, sep='\t')
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



###############################################################################################################################################

class AgGridTable:
    def __init__(
        self,
        aggrid_df=None,
        page_size=20,
        selection_mode='multiple',
        theme='streamlit',
        select_column=True,
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
        option_header=True,
        height=650,
        **kwargs
    ):
        self.aggrid_df = aggrid_df.copy() if aggrid_df is not None else pd.DataFrame()
        self.df_columns = self.aggrid_df.columns
        self.page_size = page_size
        self.selection_mode = selection_mode
        self.theme = theme
        self.select_column = select_column
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
        self.option_header = option_header
        self.extra_options = kwargs

        self.index_column = index_column
        if not self.aggrid_df.empty:
            self._prepare_dataframe()

    def format_bytes(self, size):
        units = ['Bytes', 'KB', 'MB', 'GB', 'TB']
        index = 0
        while size >= 1024 and index < len(units) - 1:
            size /= 1024
            index += 1
        return f"{size:.2f} {units[index]}"

    def _prepare_dataframe(self):
        # index_column='index' → 없으면 range(len(dataframe))으로 자동 추가
        if self.index_column == 'index':
            if 'index' not in self.aggrid_df.columns:
                self.aggrid_df.insert(0, 'index', range(len(self.aggrid_df)))

        # index_column이 None → 자동 index 추가 (기본적으로 range(len))
        elif self.index_column is None:
            if 'index' not in self.aggrid_df.columns:
                self.aggrid_df.insert(0, 'index', range(len(self.aggrid_df)))
            self.index_column = 'index'

        # index_column=False → 아무 것도 안 함
        elif self.index_column is False:
            pass

        # 다른 컬럼명일 경우 → 존재 여부 체크
        else:
            if self.index_column not in self.aggrid_df.columns:
                raise KeyError(f"'{self.index_column}' 컬럼이 DataFrame에 없습니다.")

    def update_dataframe(self, df: pd.DataFrame):
        self.aggrid_df = df.copy()
        self.df_columns = self.aggrid_df.columns
        self._prepare_dataframe()
    
    def _get_column_filters(self, default_columns=None):
        if default_columns is None:
            default_columns = list(self.df_columns)
        
        columns_filter = st.multiselect(
            "Columns",
            options=self.df_columns,
            default=default_columns,
            key="aggrid_columns_filter"
        )
        if self.index_column not in (False, None):
            columns_filter = [self.index_column] + columns_filter
        return columns_filter

    def _build_grid_options(self, dataframe, **kwargs):
        index_col_width = kwargs.get("index_col_width", self.index_col_width)
        index_min_width = kwargs.get("index_min_width", self.index_min_width)
        index_max_width = kwargs.get("index_max_width", self.index_max_width)
        index_header = kwargs.get("index_header", self.index_header)
        index_bg_color = kwargs.get("index_bg_color", self.index_bg_color)
        index_font_weight = kwargs.get("index_font_weight", self.index_font_weight)
        index_text_align = kwargs.get("index_text_align", self.index_text_align)
        min_column_width = kwargs.get("min_column_width", self.min_column_width)
        page_size = kwargs.get("page_size", self.page_size)
        selection_mode = kwargs.get("selection_mode", self.selection_mode)
        text_filter_cols = kwargs.get("text_filter_cols", self.text_filter_cols)

        gb = GridOptionsBuilder.from_dataframe(dataframe)

        if self.index_column not in (False, None):
            gb.configure_column(
                self.index_column,
                header_name=index_header,
                filter=True,
                sortable=True,
                editable=False,
                width=index_col_width,
                min_width=index_min_width,
                max_width=index_max_width,
                pinned='left',
                cellStyle={
                    "backgroundColor": index_bg_color,
                    "fontWeight": index_font_weight,
                    "padding": "0px 1px",
                    "textAlign": index_text_align
                }
            )

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
            resizable=True,
            suppressMenu=False
        )

        for col in text_filter_cols:
            if col in self.df_columns:
                gb.configure_column(col, filter="agTextColumnFilter")
        
        max_feature_len = dataframe.astype('str').map(len).max()
        for col, max_len in max_feature_len.items():
            col_width = max(self.min_column_width, max_len*8)
            gb.configure_column(col, width=col_width)

        gb.configure_selection(selection_mode=selection_mode, suppressRowDeselection=False)
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=page_size)
        gb.configure_grid_options(domLayout='normal', pivotMode=False, cellSelection=True,
                                  rowSelection=selection_mode, enableRangeSelection=True,
                                  pagination=True, paginationAutoPageSize=False,
                                  suppressColumnVirtualisation=True)
        gb.configure_side_bar(filters_panel=True, columns_panel=True)

        return gb.build()

    def _EDA_options(self, dataframe):
        with st.expander("EDA Options"):
            st_columns_1, st_columns_2 = st.columns([3,1])
            df_sumamry = DF_Summary(dataframe.drop('index', axis=1, errors='ignore'), n_samples=40)
            with st_columns_1:
                summary_table = pd.DataFrame(df_sumamry.summary).copy()
                for col in summary_table.select_dtypes(include=['object']).columns:
                    summary_table[col] = summary_table[col].astype(str)
                summary_table = summary_table.reset_index()
                st_columns_1_1, st_columns_1_2 = st.columns([9,1])
                with st_columns_1_2:
                    st_to_clipboard_button(summary_table, index=True)
                SummaryGridOptins = self._build_grid_options(summary_table, index_sortable=False, page_size=9999)
                summary_grid_response = AgGrid(summary_table, gridOptions=SummaryGridOptins,
                                               fit_columns_on_grid_load=True,
                                               columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)
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
        if dataframe is not None:
            self.update_dataframe(dataframe)

        if self.option_header:
            st_columns_1, st_columns_2, st_columns_3, st_columns_4 = st.columns([7, 0.5, 1, 1.5])

        reset_clicked = False
        if self.select_column:
            if self.option_header:
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
        else:
            columns_filter = list(self.df_columns)

        # index 컬럼이 항상 포함되도록 보장
        if self.index_column not in columns_filter and self.index_column not in (False, None):
            columns_filter = [self.index_column] + columns_filter

        if len(columns_filter) > 0:
            df_filtered = self.aggrid_df[columns_filter]
        else:
            df_filtered = self.aggrid_df.copy()

        if self.enable_EDA:
            self._EDA_options(df_filtered)

        if self.index_column not in (False, None):
            columns_sorted = [self.index_column] + \
                [col for col in columns_filter if col != self.index_column] + \
                [x for x in self.df_columns if x not in columns_filter]
        else:
            columns_sorted = columns_filter + [x for x in self.df_columns if x not in columns_filter]

        gridOptions = self._build_grid_options(self.aggrid_df[columns_sorted])
        grid_response = AgGrid(df_filtered, gridOptions=gridOptions,
                               data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                               update_mode='MODEL_CHANGED',
                               enable_enterprise_modules=self.enable_enterprise_modules,
                               height=self.height,
                               theme=self.theme,
                               **{**self.extra_options, **kwargs})
        
        df_after_select_filter = grid_response['data']
        if self.option_header:
            with st_columns_3:
                now_date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.write(" ")
                st_clipboard_download_button(df_after_select_filter, download_post_fix=now_date_str)
            with st_columns_4:
                df_memory = self.format_bytes(df_after_select_filter.memory_usage().sum())
                df_shape = df_after_select_filter.shape
                st.markdown(f"<p>· memory : {df_memory}<br>· shape : {df_shape}</p>", unsafe_allow_html=True)
            
        return grid_response

