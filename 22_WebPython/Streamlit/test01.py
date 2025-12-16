import pandas as pd
import numpy as np
from datetime import datetime

import streamlit as st
from st_aggrid import (
    AgGrid,
    GridOptionsBuilder,
    DataReturnMode,
    ColumnsAutoSizeMode,
    GridUpdateMode,
)

# DF_Summary, st_to_clipboard_button, img_to_clipboard, st_clipboard_download_button
# 은 네가 기존에 쓰던 그대로 있다고 가정


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
        session_df_key: str | None = None,   # 🔥 세션에 df 저장할 키 (예: "original_df")
        table_id: str = "aggrid",            # 🔥 UI 상태 key prefix
        **kwargs
    ):
        self.session_df_key = session_df_key
        self.table_id = table_id

        # UI 상태 key
        self.state_key_columns_filter = f"{self.table_id}_columns_filter"

        # df 초기화
        if self.session_df_key is not None and self.session_df_key in st.session_state:
            self.aggrid_df = st.session_state[self.session_df_key].copy()
        else:
            self.aggrid_df = aggrid_df.copy() if aggrid_df is not None else pd.DataFrame()
            if self.session_df_key is not None:
                st.session_state[self.session_df_key] = self.aggrid_df.copy()

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

    # ---------------------------------------------------------
    # 유틸
    # ---------------------------------------------------------
    def format_bytes(self, size):
        units = ['Bytes', 'KB', 'MB', 'GB', 'TB']
        idx = 0
        while size >= 1024 and idx < len(units) - 1:
            size /= 1024
            idx += 1
        return f"{size:.2f} {units[idx]}"

    def _prepare_dataframe(self):
        if 'index' not in self.aggrid_df.columns:
            self.aggrid_df.insert(0, 'index', self.aggrid_df.index)
        self.index_column = 'index'
        self.df_columns = self.aggrid_df.columns

    def update_dataframe(self, df: pd.DataFrame):
        self.aggrid_df = df.copy()
        self.df_columns = self.aggrid_df.columns
        self._prepare_dataframe()
        if self.session_df_key is not None:
            st.session_state[self.session_df_key] = self.aggrid_df.copy()

    # ---------------------------------------------------------
    # Columns 필터 UI
    # ---------------------------------------------------------
    def _get_column_filters(self, default_columns=None):
        if default_columns is None:
            default_columns = list(self.df_columns)

        columns_filter = st.multiselect(
            "Columns",
            options=self.df_columns,
            default=default_columns,
            key=self.state_key_columns_filter
        )

        # index 컬럼 항상 맨 앞
        if self.index_column not in columns_filter and self.index_column in self.df_columns:
            columns_filter = [self.index_column] + list(columns_filter)
        elif self.index_column in columns_filter:
            columns_filter = [self.index_column] + [
                c for c in columns_filter if c != self.index_column
            ]
        return columns_filter

    # ---------------------------------------------------------
    # GridOptions
    # ---------------------------------------------------------
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

        # index 컬럼
        if self.index_column in dataframe.columns:
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

        # 특정 컬럼 텍스트 필터
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

    # ---------------------------------------------------------
    # EDA (summary + dtype 변경)
    # ---------------------------------------------------------
    def _EDA_options(self, dataframe):
        with st.expander("EDA Options"):
            st_columns_1, st_columns_2 = st.columns([3, 1])

            # 너가 쓰던 DF_Summary 그대로 사용
            df_summary = DF_Summary(
                dataframe.drop('index', axis=1, errors='ignore'),
                n_samples=40
            )

            # ----- Summary 테이블 -----
            with st_columns_1:
                summary_table = pd.DataFrame(df_summary.summary).copy()
                summary_table = summary_table.reset_index()

                # 🔥 Arrow 에러 방지: summary_table은 표시용이니 전부 string으로
                summary_table = summary_table.astype(str)

                st_columns_1_1, st_columns_1_2 = st.columns([9, 1])
                with st_columns_1_2:
                    st_to_clipboard_button(summary_table, index=True)

                summary_gb = self._build_grid_options(
                    summary_table,
                    index_sortable=False,
                    page_size=9999
                )
                summary_gb.configure_column(
                    'dtype',
                    editable=True,
                    cellEditor='agSelectCellEditor',
                    cellEditorParams={
                        'values': ['object', 'int64', 'float64', 'bool', 'datetime']
                    }
                )
                summary_grid_options = summary_gb.build()

                summary_grid_response = AgGrid(
                    summary_table,
                    gridOptions=summary_grid_options,
                    update_mode=GridUpdateMode.MODEL_CHANGED,
                    data_return_mode=DataReturnMode.AS_INPUT,
                    enable_enterprise_modules=self.enable_enterprise_modules,
                    theme=self.theme,
                )

                # 🔘 dtype 적용 버튼
                apply_c1, apply_c2 = st.columns([8, 2])
                with apply_c2:
                    if st.button("dtype 적용", key=f"{self.table_id}_apply_dtype"):
                        edited_summary_df = pd.DataFrame(summary_grid_response["data"])
                        # index = 실제 컬럼명, dtype = 타겟 타입 (문자열)
                        if "index" in edited_summary_df.columns and "dtype" in edited_summary_df.columns:
                            for _, row in edited_summary_df.iterrows():
                                col_name = row["index"]
                                target_dtype = row["dtype"]

                                if col_name not in self.aggrid_df.columns:
                                    continue
                                if col_name == self.index_column:
                                    continue  # index 컬럼은 건드리지 않음

                                series = self.aggrid_df[col_name]

                                try:
                                    if target_dtype == "int64":
                                        self.aggrid_df[col_name] = pd.to_numeric(
                                            series, errors="coerce"
                                        ).astype("Int64")
                                    elif target_dtype == "float64":
                                        self.aggrid_df[col_name] = pd.to_numeric(
                                            series, errors="coerce"
                                        ).astype("float64")
                                    elif target_dtype == "bool":
                                        self.aggrid_df[col_name] = series.astype("bool")
                                    elif target_dtype == "datetime":
                                        self.aggrid_df[col_name] = pd.to_datetime(
                                            series, errors="coerce"
                                        )
                                    elif target_dtype == "object":
                                        self.aggrid_df[col_name] = series.astype("object")
                                except Exception as e:
                                    st.warning(f"[{col_name}] dtype 변환 중 오류: {e}")

                            # 세션 df 갱신
                            if self.session_df_key is not None:
                                st.session_state[self.session_df_key] = self.aggrid_df.copy()

                            self.df_columns = self.aggrid_df.columns
                            # 새 dtype 기준으로 전체 다시 렌더
                            st.rerun()

            # ----- Summary Plot -----
            with st_columns_2:
                if summary_grid_response["selected_rows"] is not None:
                    selected_columns = [row['index'] for row in summary_grid_response["selected_rows"]]
                    if len(selected_columns) > 0:
                        fig = df_summary.summary_plot(on=selected_columns, return_plot=True)

                        st_c1, st_c2 = st.columns([8, 2])
                        with st_c2:
                            if st.button("📋", key=f"{self.table_id}_summary_img"):
                                img_to_clipboard(fig)
                                st.toast("Complete img to clipboard!")
                        st.pyplot(fig)

    # ---------------------------------------------------------
    # render
    # ---------------------------------------------------------
    def render(self, dataframe=None, **kwargs):
        if dataframe is not None:
            self.update_dataframe(dataframe)

        # [Header] ----------------------------------------------------------
        st_columns_1, st_columns_2, st_columns_3, st_columns_4 = st.columns([7, 0.5, 1, 1.5])

        reset_clicked = False
        with st_columns_2:
            st.markdown('<p></p>', unsafe_allow_html=True)
            if st.button("🔄", key=f"{self.table_id}_reset_cols"):
                st.session_state[self.state_key_columns_filter] = list(self.df_columns)
                reset_clicked = True

        with st_columns_1:
            if reset_clicked:
                columns_filter = self._get_column_filters(default_columns=list(self.df_columns))
            else:
                columns_filter = self._get_column_filters()

        # 선택된 컬럼 없으면 전체 표시
        if len(columns_filter) > 1:
            df_filtered = self.aggrid_df[columns_filter]
        else:
            df_filtered = self.aggrid_df.copy()

        # [EDA] -------------------------------------------------------------
        if self.enable_EDA:
            self._EDA_options(df_filtered)

        # [Main Grid] --------------------------------------------------------
        columns_sorted = [self.index_column] + \
            [col for col in columns_filter if col != self.index_column] + \
            [x for x in self.df_columns if x not in columns_filter]

        grid_gb = self._build_grid_options(self.aggrid_df[columns_sorted])
        gridOptions = grid_gb.build()

        grid_response = AgGrid(
            df_filtered,
            gridOptions=gridOptions,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=False,
            enable_enterprise_modules=self.enable_enterprise_modules,
            height=self.height,
            theme=self.theme,
            **{**self.extra_options, **kwargs}
        )

        df_after_select_filter = grid_response['data']

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
