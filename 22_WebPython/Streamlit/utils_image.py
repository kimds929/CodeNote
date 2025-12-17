# utils/rect_canvas.py
from __future__ import annotations

import io
from typing import Callable, Optional, Dict, Any, Tuple, List

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image


class ImageRectAnnotator:
    """
    Streamlit drawable-canvas 기반:
    - pyplot Figure를 배경으로 깔고,
    - 사각형(1개)만 그릴 수 있게 유지,
    - Remove/Submit 버튼 제공,
    - (NEW) options만 있으면 element가 비어도 옵션 UI를 항상 렌더링
    - (NEW) element_kwargs로 초기값/조건 반영 가능
    - Submit 시 좌표 + meta(options 입력값)를 payload로 합쳐 콜백에 전달
    """

    def __init__(
        self,
        key_prefix: str,
        canvas_size: Tuple[int, int] = (500, 300),
        stroke_color: str = "red",
        stroke_width: int = 1,
        fill_color: str = "rgba(255, 0, 0, 0.1)",
        on_submit: Optional[Callable[[Dict[str, Any]], None]] = None,
        remove_label: str = "Remove Rectangle",
        submit_label: str = "Submit",
        buttons_ratio: Tuple[int, int] = (1, 1),
        signature_ndigits: int = 1,
        # options ui
        options: Optional[List[Dict[str, Any]]] = None,
        element: Optional[Dict[str, Any]] = None,
        element_kwargs: Optional[Dict[str, Any]] = None,
        options_title: str = None,
    ):
        self.key_prefix = key_prefix
        self.canvas_w, self.canvas_h = canvas_size
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.fill_color = fill_color
        self.on_submit = on_submit
        self.remove_label = remove_label
        self.submit_label = submit_label
        self.buttons_ratio = buttons_ratio
        self.signature_ndigits = signature_ndigits

        self.options = options or []

        # element + element_kwargs 병합 규칙
        element_kwargs = element_kwargs or {}
        if not element:
            self.element = dict(element_kwargs)
        else:
            self.element = dict(element)
            self.element.update(element_kwargs)

        self.options_title = options_title

        # session keys
        self._k_canvas_key = f"{key_prefix}__canvas_key"
        self._k_last_drawing = f"{key_prefix}__last_drawing"
        self._k_last_sig = f"{key_prefix}__last_sig"
        self._k_meta = f"{key_prefix}__meta"

        self._ensure_state()

    # --------------------
    # public
    # --------------------
    def render(
        self,
        fig,
        orig_size: Optional[Tuple[int, int]] = None,
        show_debug: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        fig: matplotlib pyplot figure
        orig_size: 원본 좌표계로 변환하고 싶으면 (orig_w, orig_h) 제공
                   None이면 orig_xyxy는 canvas와 동일하게 반환
        반환: submit이 눌린 경우 payload(dict), 아니면 None
        """
        bg = self._figure_to_bg_image(fig, (self.canvas_w, self.canvas_h))

        # 버튼 영역
        c1, c2 = st.columns(list(self.buttons_ratio))
        with c1:
            if st.button(self.remove_label, key=f"{self.key_prefix}__btn_remove"):
                self._reset()
                st.rerun()

        with c2:
            submit = st.button(self.submit_label, key=f"{self.key_prefix}__btn_submit")

        # ---- NEW: element가 비어도 options만 있으면 옵션 UI 렌더 ----
        if self._should_render_meta_ui():
            meta = self._render_meta_ui()
            st.session_state[self._k_meta] = meta
        else:
            st.session_state[self._k_meta] = {}

        # 캔버스
        canvas_result = st_canvas(
            fill_color=self.fill_color,
            stroke_width=self.stroke_width,
            stroke_color=self.stroke_color,
            background_image=bg,
            height=self.canvas_h,
            width=self.canvas_w,
            drawing_mode="rect",
            key=f"{self.key_prefix}__canvas_{st.session_state[self._k_canvas_key]}",
            update_streamlit=True,
            initial_drawing=st.session_state[self._k_last_drawing],
        )

        # 사각형 1개 유지
        self._keep_single_rect(canvas_result)

        # submit 처리
        payload = None
        if submit:
            payload = self._build_payload(orig_size=orig_size)
            if payload is None:
                st.warning("보낼 네모가 없어요. 먼저 네모를 그려주세요.")
            else:
                payload["meta"] = dict(st.session_state.get(self._k_meta, {}))
                if self.on_submit:
                    self.on_submit(payload)
                st.success("좌표 전송 완료!")

        if show_debug:
            st.write("last_drawing:", st.session_state[self._k_last_drawing])
            st.write("meta:", st.session_state.get(self._k_meta, {}))

        return payload

    def get_last_payload(self, orig_size: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
        payload = self._build_payload(orig_size=orig_size)
        if payload is None:
            return None
        payload["meta"] = dict(st.session_state.get(self._k_meta, {}))
        return payload

    # --------------------
    # internal
    # --------------------
    def _ensure_state(self):
        if self._k_canvas_key not in st.session_state:
            st.session_state[self._k_canvas_key] = 0
        if self._k_last_drawing not in st.session_state:
            st.session_state[self._k_last_drawing] = {"version": "4.4.0", "objects": []}
        if self._k_last_sig not in st.session_state:
            st.session_state[self._k_last_sig] = None
        if self._k_meta not in st.session_state:
            st.session_state[self._k_meta] = {}

    def _reset(self):
        st.session_state[self._k_canvas_key] += 1
        st.session_state[self._k_last_drawing] = {"version": "4.4.0", "objects": []}
        st.session_state[self._k_last_sig] = None
        st.session_state[self._k_meta] = {}

    def _figure_to_bg_image(self, fig, canvas_size: Tuple[int, int]) -> Image.Image:
        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        pil_img = Image.open(buf).convert("RGBA")
        return pil_img.resize(canvas_size, Image.Resampling.NEAREST)

    def _rect_signature(self, obj: Dict[str, Any]) -> Tuple[float, float, float, float]:
        nd = self.signature_ndigits
        left = round(float(obj.get("left", 0.0)), nd)
        top = round(float(obj.get("top", 0.0)), nd)
        w = round(float(obj.get("width", 0.0)), nd)
        h = round(float(obj.get("height", 0.0)), nd)
        return (left, top, w, h)

    def _extract_rect_xyxy(self, obj: Dict[str, Any]) -> Tuple[float, float, float, float]:
        left = float(obj.get("left", 0))
        top = float(obj.get("top", 0))
        w = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
        h = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))
        return left, top, left + w, top + h

    def _keep_single_rect(self, canvas_result):
        if not canvas_result.json_data or "objects" not in canvas_result.json_data:
            return

        objs = canvas_result.json_data["objects"]
        if not objs:
            return

        # (1) 1개면 저장만
        if len(objs) == 1:
            last = objs[0]
            sig = self._rect_signature(last)
            if st.session_state[self._k_last_sig] != sig:
                st.session_state[self._k_last_sig] = sig
                st.session_state[self._k_last_drawing] = {
                    "version": canvas_result.json_data.get("version", "4.4.0"),
                    "objects": [last],
                }
            return

        # (2) 2개 이상이면 마지막 1개만 남기고 rerun 1회
        last = objs[-1]
        sig = self._rect_signature(last)
        needs_cleanup = (
            st.session_state[self._k_last_sig] != sig
            or len(st.session_state[self._k_last_drawing].get("objects", [])) != 1
        )
        if needs_cleanup:
            st.session_state[self._k_last_sig] = sig
            st.session_state[self._k_last_drawing] = {
                "version": canvas_result.json_data.get("version", "4.4.0"),
                "objects": [last],
            }
            st.session_state[self._k_canvas_key] += 1
            st.rerun()

    def _build_payload(self, orig_size: Optional[Tuple[int, int]]) -> Optional[Dict[str, Any]]:
        objects = st.session_state[self._k_last_drawing].get("objects", [])
        if not objects:
            return None

        rect_obj = objects[0]
        x1, y1, x2, y2 = self._extract_rect_xyxy(rect_obj)

        if orig_size is None:
            orig_w, orig_h = self.canvas_w, self.canvas_h
            ox1, oy1, ox2, oy2 = x1, y1, x2, y2
        else:
            orig_w, orig_h = orig_size
            sx = orig_w / self.canvas_w
            sy = orig_h / self.canvas_h
            ox1, oy1, ox2, oy2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy

        return {
            "canvas_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "orig_xyxy": [float(ox1), float(oy1), float(ox2), float(oy2)],
            "canvas_size": [int(self.canvas_w), int(self.canvas_h)],
            "orig_size": [int(orig_w), int(orig_h)],
        }

    # --------------------
    # meta ui (NEW behavior)
    # --------------------
    def _should_render_meta_ui(self) -> bool:
        # NEW: element가 비어도 options만 있으면 무조건 렌더
        return len(self.options) > 0

    def _render_meta_ui(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}

        if self.options_title:
            st.caption(self.options_title)

        for opt in self.options:
            otype = opt.get("type")  # selectbox/radio/checkbox/multiselect/text_input
            key = opt.get("key")
            label = opt.get("label", key)
            if not otype or not key:
                continue

            widget_key = f"{self.key_prefix}__meta__{key}"

            # 우선순위: session_state(이전 입력) > element(초기값) > opt.default > None
            if widget_key in st.session_state:
                current = st.session_state[widget_key]
            else:
                current = self.element.get(key, opt.get("default", None))

            if otype == "selectbox":
                choices = opt.get("choices", [])
                if not choices:
                    meta[key] = None
                    continue
                if current not in choices:
                    current = choices[0]
                index = choices.index(current)
                meta[key] = st.selectbox(label, choices, index=index, key=widget_key)

            elif otype == "radio":
                choices = opt.get("choices", [])
                if not choices:
                    meta[key] = None
                    continue
                if current not in choices:
                    current = choices[0]
                index = choices.index(current)
                meta[key] = st.radio(
                    label,
                    choices,
                    index=index,
                    key=widget_key,
                    horizontal=bool(opt.get("horizontal", False)),
                )

            elif otype == "checkbox":
                meta[key] = st.checkbox(
                    label,
                    value=bool(current) if current is not None else False,
                    key=widget_key,
                )

            elif otype == "multiselect":
                choices = opt.get("choices", [])
                if current is None:
                    current = []
                if isinstance(current, str):
                    current = [current]
                current = [v for v in current if v in choices]
                meta[key] = st.multiselect(label, choices, default=current, key=widget_key)

            elif otype == "text_input":
                meta[key] = st.text_input(label, value=str(current) if current is not None else "", key=widget_key)

            else:
                continue

        return meta
