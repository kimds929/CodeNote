import io
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas


class ImageRectAnnotator:
    def __init__(
        self,
        key_prefix: str,
        canvas_size: Tuple[int, int] = (500, 300),
        stroke_color: str = "yellow",
        stroke_width: int = 1,
        fill_color: str = "rgba(255, 0, 0, 0.1)",
        on_submit: Optional[Callable[[Dict[str, Any]], None]] = None,
        remove_label: str = "Remove Rectangle",
        submit_label: str = "Submit",
        buttons_ratio: Tuple[int, int] = (1, 1),
        signature_ndigits: int = 1,
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
        self._k_meta = f"{key_prefix}__meta"
        self._k_init_xyxy = f"{key_prefix}__init_xyxy"
        self._k_mode_flag = f"{key_prefix}__mode_flag"  # transform 모드 여부
        self._k_canvas_result = f"{key_prefix}__canvas_result"  # 최신 canvas_result 저장
        self._k_last_canvas_result = f"{key_prefix}__last_canvas_result"  # 최신 canvas_result 저장

        self.extra_payload: Dict[str, Any] = {}
        self._ensure_state()

    def add_payload(self, extra: Dict[str, Any]) -> None:
        if not isinstance(extra, dict):
            raise ValueError("extra payload must be a dictionary")
        self.extra_payload.update(extra)

    def render(
        self,
        fig,
        orig_size: Optional[Tuple[int, int]] = None,
        show_debug: bool = False,
        canvas_size: Tuple[int, int] = [None, None],
        init_xyxy: Optional[Tuple[float, float, float, float]] = None
    ) -> Optional[Dict[str, Any]]:
        
        self.canvas_w = self.canvas_w if canvas_size[0] is None else canvas_size[0]
        self.canvas_h = self.canvas_h if canvas_size[1] is None else canvas_size[1]
        bg = self._figure_to_bg_image(fig, (self.canvas_w, self.canvas_h))

        # 버튼 영역
        c1, c2 = st.columns(list(self.buttons_ratio))
        with c1:
            if st.button(self.remove_label, key=f"{self.key_prefix}__btn_remove"):
                self._reset()
                st.rerun()

        with c2:
            submit = st.button(self.submit_label, key=f"{self.key_prefix}__btn_submit")
            # submit = st.button(self.submit_label, key=False)
            

        # 옵션 UI 렌더링
        if self._should_render_meta_ui():
            meta = self._render_meta_ui()
            st.session_state[self._k_meta] = meta
        else:
            st.session_state[self._k_meta] = {}

        
            
        # init_xyxy 좌표설정
        initial_drawing = st.session_state[self._k_last_drawing]
        if len(initial_drawing['objects']) == 0:       # 처음 render하는 경우
            if (init_xyxy is not None) and (len(init_xyxy) == 4):
                x1, y1, x2, y2 = init_xyxy
                init_xyxy_options = {"type": "rect", "left": int(x1), "top": int(y1), "width": int(x2-x1), "height": int(y2-y1), 
                "fill": self.fill_color, "stroke": "yellow", "strokeWidth": 1, "angle": 0, "scaleX": 1, "scaleY": 1,
                "strokeLineCap":"butt", "strokeDashOffset":0, "strokeLineJoin":"miter", 
                "strokeMiterLimit":4, "scaleX":1, "scaleY":1, "angle":0, "opacity":1, "backgroundColor":"", 
                "fillRule":"nonzero", "paintFirst":"fill", "globalCompositeOperation":"source-over", "skewX":0, "skewY":0, "rx":0, "ry":0
                }
                initial_drawing['objects'] = [init_xyxy_options]
                st.session_state[self._k_mode_flag] = True  # transform
            else:
                st.session_state[self._k_mode_flag] = False
        
        # drawing_mode 및 update_streamlit 설정
        if st.session_state[self._k_mode_flag]:
            drawing_mode = "transform"
        else:
            drawing_mode = "rect"
        update_streamlit = drawing_mode != 'transform'   

        # 캔버스 렌더링
        canvas_result = st_canvas(
            fill_color=self.fill_color,
            stroke_width=self.stroke_width,
            stroke_color=self.stroke_color,
            background_image=bg,
            height=self.canvas_h,
            width=self.canvas_w,
            drawing_mode=drawing_mode,
            key=f"{self.key_prefix}__canvas_{st.session_state[self._k_canvas_key]}",
            update_streamlit=update_streamlit,
            initial_drawing=initial_drawing,
        )

        # 현재 그린 내용 저장 (transform 모드에서도 최신 상태 반영)
        if canvas_result.json_data and "objects" in canvas_result.json_data:
            st.session_state[self._k_last_drawing] = {
                "version": canvas_result.json_data.get("version", "4.4.0"),
                "objects": canvas_result.json_data["objects"],
            }
        
        # 사각형 존재 여부 체크 (캔버스 그린 직후 판단)
        has_rect = bool(canvas_result.json_data and canvas_result.json_data.get("objects"))
        if not submit:
            if has_rect and not st.session_state[self._k_mode_flag]:
                st.session_state[self._k_mode_flag] = True
                st.rerun()
            elif not has_rect and st.session_state[self._k_mode_flag]:
                st.session_state[self._k_mode_flag] = False
        
        payload = None
        
        if submit:
            # 여기서 최신 canvas_result를 직접 사용
            st.session_state[self._k_last_canvas_result] = canvas_result.json_data
            json_data = st.session_state.get(self._k_last_canvas_result)
            
            payload = self._build_payload_from_canvas_result(json_data, orig_size=orig_size)
            if payload is None:
                st.warning("보낼 네모가 없어요. 먼저 네모를 그려주세요.")
            else:
                payload["meta"] = dict(st.session_state.get(self._k_meta, {}))
                payload.update(self.extra_payload)
                if self.on_submit:
                    self.on_submit(payload)
                st.success("Submit Success!")

        if show_debug:
            st.write("last_drawing:", st.session_state[self._k_last_drawing])
            st.write("meta:", st.session_state.get(self._k_meta, {}))
            st.write("extra_payload:", self.extra_payload)
            st.write("mode_flag:", st.session_state[self._k_mode_flag])
            st.write("drawing_mode:", drawing_mode)

        return payload

    def get_last_payload(self, orig_size: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
        json_data = st.session_state.get(self._k_last_canvas_result)
        payload = self._build_payload_from_canvas_result(json_data, orig_size=orig_size)
        if payload is None:
            return None
        payload["meta"] = dict(st.session_state.get(self._k_meta, {}))
        payload.update(self.extra_payload)
        return payload

    def _ensure_state(self):
        if self._k_canvas_key not in st.session_state:      # _k_canvas_key
            st.session_state[self._k_canvas_key] = 0
        if self._k_last_drawing not in st.session_state:    # _k_last_drawing
            st.session_state[self._k_last_drawing] = {"version": "4.4.0", "objects": []}
        if self._k_meta not in st.session_state:            # _k_meta
            st.session_state[self._k_meta] = {}
        if self._k_init_xyxy not in st.session_state:            # _k_init_xyxy
            st.session_state[self._k_init_xyxy] = False
        if self._k_mode_flag not in st.session_state:       # _k_mode_flag
            st.session_state[self._k_mode_flag] = False
        if self._k_canvas_result not in st.session_state:  # _k_canvas_result
            st.session_state[self._k_canvas_result] = None
        if self._k_last_canvas_result not in st.session_state:  # _k_last_canvas_result
            st.session_state[self._k_last_canvas_result] = None

    def _reset(self):
        st.session_state[self._k_canvas_key] += 1
        st.session_state[self._k_last_drawing] = {"version": "4.4.0", "objects": []}
        st.session_state[self._k_meta] = {}
        st.session_state[self._k_init_xyxy] = False
        st.session_state[self._k_mode_flag] = False
        st.session_state[self._k_canvas_result] = None
        st.session_state[self._k_last_canvas_result] = None

    def _figure_to_bg_image(self, fig, canvas_size: Tuple[int, int]) -> Image.Image:
        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        pil_img = Image.open(buf).convert("RGBA")
        return pil_img.resize(canvas_size, Image.Resampling.NEAREST)

    def _extract_rect_xyxy(self, obj: Dict[str, Any]) -> Tuple[float, float, float, float]:
        left = float(obj.get("left", 0))
        top = float(obj.get("top", 0))
        w = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
        h = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))
        return left, top, left + w, top + h

    def _build_payload_from_canvas_result(self, json_data, orig_size: Optional[Tuple[int, int]]) -> Optional[Dict[str, Any]]:
        if not json_data or "objects" not in json_data or not json_data["objects"]:
            return None

        rect_obj = json_data["objects"][0]
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

    def _should_render_meta_ui(self) -> bool:
        return len(self.options) > 0

    def _render_meta_ui(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if self.options_title:
            st.caption(self.options_title)

        for opt in self.options:
            otype = opt.get("type")
            key = opt.get("key")
            label = opt.get("label", key)
            if not otype or not key:
                continue

            widget_key = f"{self.key_prefix}__meta__{key}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = self.element.get(key, opt.get("default", None))
            current = st.session_state[widget_key]

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

        st.session_state[self._k_meta] = meta
        return meta
