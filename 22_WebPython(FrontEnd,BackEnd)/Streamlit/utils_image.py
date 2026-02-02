import base64
import io
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components


class ImageRectAnnotator:
    """
    components.html 값 반환 불가 환경용(query param 우회):
    - JS가 parent URL query param에 좌표 JSON을 기록
    - Python은 Submit 시 query param에서 좌표를 읽어 payload로 on_submit 호출

    추가 기능:
    - init_xyxy=None이면 초기 박스 표시/기록 X
    - hover 커서: resize / grab / crosshair
    - add_payload(extra_payload): render 전에 호출하여 submit payload에 extra_payload 포함
    - submit payload를 st.session_state에 저장
    - @st.fragment: options UI 변경 시 캔버스 영향 최소화
    """

    def __init__(
        self,
        key_prefix: str,
        canvas_size: Tuple[int, int] = (500, 300),
        stroke_color: str = "yellow",
        fill_color: str = "rgba(255, 255, 0, 0.2)",
        on_submit: Optional[Callable[[Dict[str, Any]], None]] = None,
        submit_label: str = "Submit",
        reset_label: str = "Reset",
        options: Optional[List[Dict[str, Any]]] = None,
        element_kwargs: Optional[Dict[str, Any]] = None,
        min_box_size: float = 1.0,
    ):
        self.key_prefix = key_prefix
        self.canvas_w, self.canvas_h = canvas_size
        self.stroke_color = stroke_color
        self.fill_color = fill_color
        self.on_submit = on_submit
        self.submit_label = submit_label
        self.reset_label = reset_label
        self.options = options or []
        self.element_kwargs = element_kwargs or {}
        self.min_box_size = float(min_box_size)

        # JS가 ?{_qp_key}=[x1,y1,x2,y2] 형태로 좌표를 기록
        self._qp_key = f"{key_prefix}_xyxy"

        # session_state keys
        self._k_init = f"{key_prefix}__init"  # is init?
        self._k_meta = f"{key_prefix}__meta"  # options draft
        self._k_meta_committed = f"{key_prefix}__meta_committed"  # submit 시점 확정
        self._k_extra = f"{key_prefix}__extra_payload"
        self._k_submitted = f"{key_prefix}__submitted_payload"
        self._k_opt_ver = f"{key_prefix}__opt_ver"
        self._ensure_state()

    # ------------------------- state -------------------------
    def _ensure_state(self) -> None:
        if self._k_init not in st.session_state:
            st.session_state[self._k_init] = True
        if self._k_meta not in st.session_state:
            st.session_state[self._k_meta] = {}
        if self._k_meta_committed not in st.session_state:
            st.session_state[self._k_meta_committed] = {}
        if self._k_extra not in st.session_state:
            st.session_state[self._k_extra] = {}
        if self._k_opt_ver not in st.session_state:
            st.session_state[self._k_opt_ver] = 0

    # ------------------------- extra payload -------------------------
    def add_payload(self, extra_payload: Optional[Dict[str, Any]] = None) -> None:
        """
        render 전에 호출해서 submit payload에 포함시키기:
          ann.add_payload({"study_id": "...", "user": "..."})
        """
        if extra_payload is None:
            extra_payload = {}
        if not isinstance(extra_payload, dict):
            raise TypeError("extra_payload는 dict 여야 합니다.")

        cur = st.session_state.get(self._k_extra, {}) or {}
        merged = dict(cur)
        merged.update(extra_payload)  # 새 값 우선
        st.session_state[self._k_extra] = merged

    # ------------------------- utils -------------------------
    def _fig_to_base64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="PNG", bbox_inches="tight", pad_inches=0)
        return base64.b64encode(buf.getvalue()).decode()

    def _render_options_ui(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if not self.options:
            return meta

        for spec in self.options:
            wtype = spec.get("type")
            k = spec.get("key")
            lbl = spec.get("label", k)
            if not wtype or not k:
                continue

            kwargs = {}
            kwargs.update(self.element_kwargs.get(k, {}))
            widget_key = f"{self.key_prefix}__opt__{k}"

            if wtype == "selectbox":
                choices = spec.get("choices", [])
                default = spec.get("default", 0)
                meta[k] = st.selectbox(
                    lbl,
                    choices,
                    index=default if isinstance(default, int) and 0 <= default < len(choices) else 0,
                    key=widget_key,
                    **kwargs,
                )
            elif wtype == "radio":
                choices = spec.get("choices", [])
                default = spec.get("default", 0)
                meta[k] = st.radio(
                    lbl,
                    choices,
                    index=default if isinstance(default, int) and 0 <= default < len(choices) else 0,
                    key=widget_key,
                    **kwargs,
                )
            elif wtype == "checkbox":
                meta[k] = st.checkbox(lbl, value=bool(spec.get("default", False)), key=widget_key, **kwargs)
            elif wtype == "multiselect":
                choices = spec.get("choices", [])
                default = spec.get("default", [])
                meta[k] = st.multiselect(
                    lbl,
                    choices,
                    default=default if isinstance(default, list) else [],
                    key=widget_key,
                    **kwargs,
                )
            elif wtype == "text_input":
                meta[k] = st.text_input(lbl, value=str(spec.get("default", "")), key=widget_key, **kwargs)

        return meta

    def _render_options_fragment(self) -> Dict[str, Any]:
        """
        options 값 변경으로 인한 rerun을 캔버스 영역까지 번지지 않게 하려는 의도.
        (Streamlit 동작 특성상 전체 스크립트가 재실행되더라도 fragment가 캐시/격리 효과를 줌)
        """
        @st.fragment
        def _frag():
            meta = self._render_options_ui()
            st.session_state[self._k_meta] = meta  # draft 저장
            return meta

        return _frag()

    # ------------------------- query params -------------------------
    def _get_raw_query_value(self) -> Optional[str]:
        try:
            qp = st.query_params
            return qp.get(self._qp_key, None)
        except Exception:
            qp = st.experimental_get_query_params()
            raw_list = qp.get(self._qp_key, None)
            return raw_list[0] if isinstance(raw_list, list) and raw_list else None

    def _get_xyxy_from_query_params(self) -> Optional[List[float]]:
        raw = self._get_raw_query_value()
        if not raw:
            return None
        try:
            arr = json.loads(raw)
            if not (isinstance(arr, list) and len(arr) == 4):
                return None
            x1, y1, x2, y2 = map(float, arr)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            return [x1, y1, x2, y2]
        except Exception:
            return None

    def _clear_query_param(self) -> None:
        try:
            if self._qp_key in st.query_params:
                del st.query_params[self._qp_key]
        except Exception:
            qp = st.experimental_get_query_params()
            if self._qp_key in qp:
                qp.pop(self._qp_key, None)
                st.experimental_set_query_params(**qp)

    def _reset_options_widgets(self) -> None:
        # options에 정의된 위젯 key들을 session_state에서 제거
        for spec in (self.options or []):
            k = spec.get("key")
            if not k:
                continue
            widget_key = f"{self.key_prefix}__opt__{k}"
            if widget_key in st.session_state:
                del st.session_state[widget_key]
            
    def _reset(self)->None:
        st.session_state[self._k_init] = True
        st.session_state[self._k_meta] = {}
        st.session_state[self._k_meta_committed] = {}
        st.session_state[self._k_extra] = {}
        self._reset_options_widgets()
        self._clear_query_param()
    
    # ------------------------- render -------------------------
    def render(
        self,
        fig,
        orig_size: Optional[Tuple[int, int]] = None,
        init_xyxy: Optional[Tuple[float, float, float, float]] = None,
        fit_to_canvas: bool = True,
        show_debug: bool = False,
    ) -> Optional[Dict[str, Any]]:
        # options UI (fragment)
        _ = self._render_options_fragment()              

        # 초기 박스 결정: query param 우선, 없으면 init_xyxy
        qp_xyxy = self._get_xyxy_from_query_params()
        current_coords = qp_xyxy if qp_xyxy is not None else init_xyxy

        has_init = "true" if current_coords is not None else "false"
        if current_coords is None:
            ix1, iy1, ix2, iy2 = (0, 0, 0, 0)
        else:
            ix1, iy1, ix2, iy2 = current_coords

        if st.session_state[self._k_init]:
            self._clear_query_param()
            st.session_state[self._k_init] = False
            
        # 3) 캔버스 HTML/JS (✅ dbg element 제거)
        img_b64 = self._fig_to_base64(fig)

        canvas_html = f"""
<div style="display:inline-block; user-select:none;">
  <!-- Canvas only (debug div removed) -->
  <canvas id="{self.key_prefix}_canvas" style="border:1px solid #ccc; cursor:crosshair; touch-action:none;"></canvas>
</div>

<script>
(function() {{
  // ====== DOM / Canvas ======
  const canvas = document.getElementById("{self.key_prefix}_canvas");
  const ctx = canvas.getContext("2d");

  // ====== Background Image ======
  const img = new Image();
  img.src = "data:image/png;base64,{img_b64}";

  // ====== Params injected by Python ======
  const CANVAS_W = {self.canvas_w};
  const CANVAS_H = {self.canvas_h};
  const FIT_TO_CANVAS = {str(bool(fit_to_canvas)).lower()};
  const MIN_BOX = {float(self.min_box_size)};
  const QP_KEY = {json.dumps(self._qp_key)};

  // ====== Initial rect state ======
  let hasBox = {has_init};
  let rect = {{ x1:{ix1}, y1:{iy1}, x2:{ix2}, y2:{iy2} }};

  // mode: draw | move | resize
  let mode = null;
  let activeHandle = null;
  let moveOffsetX = 0, moveOffsetY = 0;

  const HANDLE = 8, HIT = 10;

  function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}

  function normalize(r) {{
    let x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2;
    if (x1>x2) {{ const t=x1; x1=x2; x2=t; }}
    if (y1>y2) {{ const t=y1; y1=y2; y2=t; }}
    r.x1=x1; r.y1=y1; r.x2=x2; r.y2=y2;
    return r;
  }}

  function clampRect(r) {{
    normalize(r);
    r.x1 = clamp(r.x1, 0, CANVAS_W);
    r.x2 = clamp(r.x2, 0, CANVAS_W);
    r.y1 = clamp(r.y1, 0, CANVAS_H);
    r.y2 = clamp(r.y2, 0, CANVAS_H);
    normalize(r);
    return r;
  }}

  function okSize(r) {{
    normalize(r);
    return ((r.x2-r.x1) >= MIN_BOX && (r.y2-r.y1) >= MIN_BOX);
  }}

  function drawHandle(x,y) {{
    ctx.fillStyle="white"; ctx.strokeStyle="black";
    ctx.fillRect(x-HANDLE/2,y-HANDLE/2,HANDLE,HANDLE);
    ctx.strokeRect(x-HANDLE/2,y-HANDLE/2,HANDLE,HANDLE);
  }}

  function getHandle(mx,my) {{
    if (!hasBox) return null;
    const x1=rect.x1, y1=rect.y1, x2=rect.x2, y2=rect.y2;
    if (Math.abs(mx-x1)<HIT && Math.abs(my-y1)<HIT) return "tl";
    if (Math.abs(mx-x2)<HIT && Math.abs(my-y1)<HIT) return "tr";
    if (Math.abs(mx-x1)<HIT && Math.abs(my-y2)<HIT) return "bl";
    if (Math.abs(mx-x2)<HIT && Math.abs(my-y2)<HIT) return "br";
    return null;
  }}

  function inside(mx,my) {{
    if (!hasBox) return false;
    const x1=Math.min(rect.x1,rect.x2), x2=Math.max(rect.x1,rect.x2);
    const y1=Math.min(rect.y1,rect.y2), y2=Math.max(rect.y1,rect.y2);
    return (mx>x1 && mx<x2 && my>y1 && my<y2);
  }}

  function draw() {{
    ctx.clearRect(0,0,CANVAS_W,CANVAS_H);
    if (FIT_TO_CANVAS) ctx.drawImage(img,0,0,CANVAS_W,CANVAS_H);
    else ctx.drawImage(img,0,0);

    if (hasBox) {{
      normalize(rect);
      const w=rect.x2-rect.x1, h=rect.y2-rect.y1;
      ctx.strokeStyle="{self.stroke_color}";
      ctx.lineWidth=2;
      ctx.strokeRect(rect.x1,rect.y1,w,h);
      ctx.fillStyle="{self.fill_color}";
      ctx.fillRect(rect.x1,rect.y1,w,h);

      drawHandle(rect.x1,rect.y1);
      drawHandle(rect.x2,rect.y1);
      drawHandle(rect.x1,rect.y2);
      drawHandle(rect.x2,rect.y2);
    }}
  }}

  // ✅ write coords into parent URL query param
  function writeQueryParam(valueArr) {{
    try {{
      const u = new URL(window.parent.location.href);
      u.searchParams.set(QP_KEY, JSON.stringify(valueArr));
      window.parent.history.replaceState({{}}, "", u.toString());
    }} catch(e) {{
      // parent 접근 실패 시 조용히 무시
    }}
  }}

  function save() {{
    normalize(rect);
    clampRect(rect);
    // if (!okSize(rect)) return;
    const v=[rect.x1,rect.y1,rect.x2,rect.y2];
    writeQueryParam(v);
  }}

  function getMouse(e) {{
    const b=canvas.getBoundingClientRect();
    const mx=(e.clientX-b.left) * (CANVAS_W/b.width);
    const my=(e.clientY-b.top) * (CANVAS_H/b.height);
    return {{x:mx,y:my}};
  }}

  function updateCursor(mx, my) {{
    if (!hasBox) {{ canvas.style.cursor="crosshair"; return; }}
    const h = getHandle(mx, my);
    if (h === "tl" || h === "br") {{ canvas.style.cursor="nwse-resize"; return; }}
    if (h === "tr" || h === "bl") {{ canvas.style.cursor="nesw-resize"; return; }}
    if (inside(mx, my)) {{
      canvas.style.cursor = (mode === "move") ? "grabbing" : "grab";
      return;
    }}
    canvas.style.cursor="crosshair";
  }}

  img.onload = () => {{
    canvas.width=CANVAS_W;
    canvas.height=CANVAS_H;

    if (hasBox) clampRect(rect);
    draw();

    // init_xyxy가 있을 때만 최초 저장
    if (hasBox) save();
  }};

  canvas.addEventListener("pointerdown", (e) => {{
    canvas.setPointerCapture(e.pointerId);
    const m=getMouse(e); const mx=m.x, my=m.y;

    if (!hasBox) {{
      mode="draw";
      rect.x1=mx; rect.y1=my; rect.x2=mx; rect.y2=my;
      hasBox=true;
      updateCursor(mx,my);
      draw();
      return;
    }}

    const h=getHandle(mx,my);
    if (h) {{ mode="resize"; activeHandle=h; updateCursor(mx,my); return; }}

    if (inside(mx,my)) {{
      mode="move";
      moveOffsetX=mx-rect.x1; moveOffsetY=my-rect.y1;
      updateCursor(mx,my);
      return;
    }}

    mode="draw";
    rect.x1=mx; rect.y1=my; rect.x2=mx; rect.y2=my;
    hasBox=true;
    updateCursor(mx,my);
    draw();
  }});

  canvas.addEventListener("pointermove", (e) => {{
    const m=getMouse(e); const mx=m.x, my=m.y;

    if (!mode) {{ updateCursor(mx,my); return; }}

    if (mode==="draw") {{
      rect.x2=mx; rect.y2=my;
      clampRect(rect);
      updateCursor(mx,my);
      draw();
      return;
    }}

    if (mode==="move") {{
      const w=rect.x2-rect.x1, h=rect.y2-rect.y1;
      rect.x1=mx-moveOffsetX; rect.y1=my-moveOffsetY;
      rect.x2=rect.x1+w; rect.y2=rect.y1+h;

      const dx1=0-rect.x1, dy1=0-rect.y1;
      const dx2=CANVAS_W-rect.x2, dy2=CANVAS_H-rect.y2;
      if (rect.x1<0) {{ rect.x1+=dx1; rect.x2+=dx1; }}
      if (rect.y1<0) {{ rect.y1+=dy1; rect.y2+=dy1; }}
      if (rect.x2>CANVAS_W) {{ rect.x1+=dx2; rect.x2+=dx2; }}
      if (rect.y2>CANVAS_H) {{ rect.y1+=dy2; rect.y2+=dy2; }}

      updateCursor(mx,my);
      draw();
      return;
    }}

    if (mode==="resize") {{
      if (activeHandle==="br") {{ rect.x2=mx; rect.y2=my; }}
      else if (activeHandle==="tl") {{ rect.x1=mx; rect.y1=my; }}
      else if (activeHandle==="tr") {{ rect.x2=mx; rect.y1=my; }}
      else if (activeHandle==="bl") {{ rect.x1=mx; rect.y2=my; }}

      clampRect(rect);
      updateCursor(mx,my);
      draw();
      return;
    }}
  }});

  canvas.addEventListener("pointerup", (e) => {{
    if (!mode) return;
    mode=null; activeHandle=null;
    try {{ canvas.releasePointerCapture(e.pointerId); }} catch(err) {{}}

    const m=getMouse(e); updateCursor(m.x, m.y);
    save();
  }});

}})();
</script>
"""     
        components.html(canvas_html, height=self.canvas_h + 5, width=self.canvas_w + 5)
        
        # Rest / Submit 버튼
        colb1, colb2 = st.columns([1, 1])
        with colb1:
            reset_clicked = st.button(self.reset_label, key=f"{self.key_prefix}_reset_bottom")
        with colb2:
            submit_clicked = st.button(self.submit_label, key=f"{self.key_prefix}_submit_bottom", type="primary")

        if reset_clicked:
            self._reset()
            print('d')
            st.rerun()

        # Python 디버그만 유지
        if show_debug:
            st.write("DEBUG query_raw:", self._get_raw_query_value())
            st.write("DEBUG query_xyxy:", self._get_xyxy_from_query_params())
            st.write("DEBUG meta(draft):", st.session_state.get(self._k_meta))
            st.write("DEBUG meta(committed):", st.session_state.get(self._k_meta_committed))
            st.write("DEBUG extra_payload:", st.session_state.get(self._k_extra))

        # 5) submit
        if submit_clicked:
            xyxy_raw  = self._get_xyxy_from_query_params()
            if not xyxy_raw :
                st.warning("No Box in Canvas.")
                return None

            
            # submit좌표
            x1, y1, x2, y2 = xyxy_raw 
            
            if (x2 - x1) < self.min_box_size or (y2 - y1) < self.min_box_size:
                st.warning(f"Too small Box. Minimum size Box : {self.min_box_size}px")
                return None
            
            orig_w, orig_h = orig_size if orig_size else (self.canvas_w, self.canvas_h)
            sx, sy = orig_w / self.canvas_w, orig_h / self.canvas_h
            
            # submit 시점 meta 확정(commit)
            st.session_state[self._k_meta_committed] = dict(st.session_state.get(self._k_meta, {}) or {})
            committed_meta = st.session_state[self._k_meta_committed]

            payload = {
                "canvas_xyxy": [x1, y1, x2, y2],
                "orig_xyxy": [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                "orig_size": [orig_w, orig_h],
                "canvas_size": [self.canvas_w, self.canvas_h],
                "meta": committed_meta,
                "extra_payload": st.session_state.get(self._k_extra, {}) or {},
            }

            st.session_state[self._k_submitted] = payload

            if self.on_submit:
                self.on_submit(payload)

            st.success("Submit Success!")
            return payload

        return None
