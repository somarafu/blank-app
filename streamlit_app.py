import json
import html
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Digital Heritage Guide",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# 2. DATA
#    GitHub 구조:
#    streamlit_app.py
#    data/
#      ├─ heritage.json
#      └─ tickets.json
# =========================================================
@st.cache_data
def load_data():
    heritage_path = BASE_DIR / "data" / "heritage.json"
    ticket_path = BASE_DIR / "data" / "tickets.json"

    missing = [str(p.relative_to(BASE_DIR)) for p in [heritage_path, ticket_path] if not p.exists()]
    if missing:
        st.error(
            "필수 데이터 파일을 찾을 수 없습니다.\n\n"
            + "\n".join(f"- `{m}`" for m in missing)
            + "\n\nGitHub 저장소에서 `data` 폴더 안에 JSON 파일이 있는지 확인해 주세요."
        )
        st.stop()

    heritage = json.loads(heritage_path.read_text(encoding="utf-8"))
    tickets = json.loads(ticket_path.read_text(encoding="utf-8"))
    return heritage, tickets


HERITAGE, TICKETS = load_data()
HERITAGE_BY_ID = {item["id"]: item for item in HERITAGE}

LANGS = {
    "한국어": ("ko", "ko-KR"),
    "English": ("en", "en-US"),
    "Ελληνικά": ("el", "el-GR"),
}


# =========================================================
# 3. SESSION STATE
# =========================================================
def init_state():
    defaults = {
        "authenticated": False,
        "ticket_no": None,
        "saved": set(),
        "viewed": [],
        "selected_id": None,
        "lang_label": "한국어",
        "login_code": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# =========================================================
# 4. DESIGN
#    2번째 참고 이미지처럼:
#    white / pale blue / royal blue / soft gray
# =========================================================
st.markdown(
    """
<style>
/* ---------- COLOR SYSTEM ---------- */
:root {
    --page: #F8F9FC;
    --surface: #FFFFFF;
    --surface-soft: #F5F7FC;
    --blue: #4C61B8;
    --blue-deep: #344A9B;
    --blue-soft: #EEF2FC;
    --line: #E1E5EF;
    --line-blue: #CAD5F2;
    --text: #171A22;
    --subtext: #5F6470;
    --muted: #8A8F99;
}

/* ---------- GLOBAL ---------- */
html {
    color-scheme: light !important;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--page) !important;
    color: var(--text) !important;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

/* 모든 기본 본문 텍스트가 흰색으로 바뀌지 않도록 강제 */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
.stCaption,
label {
    color: var(--text) !important;
}

/* ---------- INPUT ---------- */
div[data-baseweb="input"] > div {
    background: #FFFFFF !important;
    border: 1px solid #D7DCE7 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

div[data-baseweb="input"] input {
    color: #111318 !important;
    -webkit-text-fill-color: #111318 !important;
    background: #FFFFFF !important;
    font-size: 16px !important;
}

div[data-baseweb="input"] input::placeholder {
    color: #969BA6 !important;
    -webkit-text-fill-color: #969BA6 !important;
}

/* ---------- SELECT ---------- */
div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #111318 !important;
    border-color: #DDE2EC !important;
    border-radius: 10px !important;
}

/* ---------- BUTTON ---------- */
div[data-testid="stButton"] button {
    min-height: 44px;
    border-radius: 11px;
    font-weight: 750;
    transition: .15s ease;
}

/* 일반 버튼 */
div[data-testid="stButton"] button:not([kind="primary"]) {
    background: #FFFFFF !important;
    color: #20242E !important;
    border: 1px solid #D7DDE9 !important;
}

div[data-testid="stButton"] button:not([kind="primary"]) p {
    color: #20242E !important;
}

/* 주요 버튼 */
div[data-testid="stButton"] button[kind="primary"] {
    background: var(--blue) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--blue) !important;
}

div[data-testid="stButton"] button[kind="primary"] p {
    color: #FFFFFF !important;
}

div[data-testid="stButton"] button:hover {
    transform: translateY(-1px);
}

/* ---------- LOGIN ---------- */
.login-browser {
    background: #FFFFFF;
    border: 1px solid #E2E5EB;
    border-radius: 24px 24px 0 0;
    height: 46px;
    padding: 0 18px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.dot {
    width: 11px;
    height: 11px;
    border-radius: 999px;
    display: inline-block;
}
.dot-red { background:#CA7765; }
.dot-yellow { background:#D3B76A; }
.dot-green { background:#80B46A; }

.login-heading {
    color: #171A22;
    font-family: Georgia, "Times New Roman", "Noto Serif KR", serif;
    font-weight: 800;
    font-size: 34px;
    line-height: 1.22;
    margin: 8px 0 10px 0;
}

.login-eyebrow {
    color: var(--blue);
    font-weight: 800;
    font-size: 13px;
    letter-spacing: .12em;
}

.login-description {
    color: #4D525C;
    line-height: 1.75;
    font-size: 15px;
    margin-bottom: 7px;
}

.login-note {
    background: var(--blue-soft);
    color: #333A49;
    border: 1px solid #DDE6FB;
    border-radius: 12px;
    padding: 12px 14px;
    font-size: 13px;
    line-height: 1.55;
    margin-top: 5px;
}

/* Streamlit border container: 로그인 카드 */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-color: #E1E5EC !important;
    border-radius: 0 0 24px 24px !important;
}

/* ---------- MAIN HEADER ---------- */
.platform-shell {
    background: #FFFFFF;
    border: 1px solid #E1E4EC;
    border-radius: 20px;
    box-shadow: 0 14px 42px rgba(45, 54, 78, .07);
    overflow: hidden;
    margin-bottom: 16px;
}

.browser-top {
    height: 40px;
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 0 16px;
    border-bottom: 1px solid #ECEEF3;
}

.platform-title-row {
    padding: 15px 20px 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.platform-brand {
    color: #1C2D5D;
    font-size: 22px;
    font-family: Georgia, "Times New Roman", serif;
    font-weight: 800;
}

.route-pill {
    background: #F0F3FB;
    color: #445681;
    border: 1px solid #E0E5F1;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}

.verified {
    margin: 0 20px 18px;
    background: linear-gradient(90deg,#F1F4FC,#FAFBFE);
    border: 1px solid #DCE3F3;
    border-radius: 14px;
    padding: 16px;
    display: flex;
    gap: 13px;
    align-items: center;
}

.verified-icon {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    background: var(--blue);
    color: white;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 21px;
    flex: 0 0 auto;
}

.verified-title {
    color: var(--blue-deep);
    font-weight: 800;
    font-size: 17px;
}

.verified-sub {
    color: #3E434D;
    margin-top: 3px;
    font-size: 13px;
}

/* ---------- SIDE PANELS ---------- */
.side-panel {
    background: #FFFFFF;
    border: 1px solid #E1E5ED;
    border-radius: 18px;
    padding: 18px 16px;
    min-height: 500px;
}

.side-title {
    color: var(--blue);
    text-align: center;
    font-size: 19px;
    font-weight: 850;
    padding-bottom: 13px;
    border-bottom: 1px solid #DDE3EE;
    margin-bottom: 7px;
}

.side-row {
    padding: 15px 3px;
    border-bottom: 1px solid #ECEEF3;
}

.side-row:last-child {
    border-bottom: 0;
}

.side-row-title {
    color: #3151A1;
    font-weight: 800;
    font-size: 15px;
}

.side-row-text {
    color: #555B65;
    font-size: 12px;
    line-height: 1.55;
    margin-top: 4px;
}

/* ---------- HERITAGE ---------- */
.section-label {
    color: #344F9F;
    font-weight: 850;
    font-size: 20px;
    margin: 4px 0 12px;
}

.heritage-card {
    background: #FFFFFF;
    border: 1px solid #E0E4EC;
    border-radius: 14px;
    padding: 14px;
    min-height: 260px;
}

.card-title {
    color: #4560B0;
    font-size: 18px;
    font-weight: 850;
}

.card-subtitle {
    color: #373C45;
    font-size: 12.5px;
    line-height: 1.5;
    min-height: 40px;
    margin-top: 4px;
}

.card-visual {
    height: 130px;
    background: linear-gradient(135deg,#F0F3FA,#F8F6F2);
    border: 1px solid #E5E8EE;
    border-radius: 11px;
    margin: 11px 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 62px;
}

.card-meta {
    color: #7A808A;
    font-size: 11px;
}

.detail-box {
    background: #FFFFFF;
    border: 1px solid #DCE3F2;
    border-radius: 15px;
    padding: 18px;
    margin-top: 14px;
}

.detail-title {
    color: #314D99;
    font-weight: 850;
    font-size: 21px;
}

.detail-meta {
    color: #808690;
    font-size: 12px;
    margin-top: 3px;
}

.detail-text {
    color: #252931;
    font-size: 14px;
    line-height: 1.75;
    margin-top: 10px;
}

.saved-chip {
    background: #FFFFFF;
    border: 1px solid #E0E4EC;
    border-radius: 11px;
    padding: 10px;
    min-height: 65px;
}

.saved-title {
    color:#20242D;
    font-weight:750;
    font-size:13px;
}

.saved-meta {
    color:#8A8F98;
    font-size:10.5px;
    margin-top:3px;
}

.quote-box {
    background: #F0F4FC;
    border: 1px solid #DCE5F8;
    color: #405CA8;
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
    font-size: 15px;
    font-weight: 750;
    margin-top: 16px;
}

/* ---------- TABS ---------- */
div[data-baseweb="tab-list"] {
    gap: 8px;
}

button[data-baseweb="tab"] {
    color: #5B6170 !important;
    font-weight: 750 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #344F9F !important;
}

/* ---------- RESPONSIVE ---------- */
@media (max-width: 900px) {
    .side-panel { min-height: auto; }
    .login-heading { font-size: 28px; }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 5. HELPERS
# =========================================================
def tr(item, field):
    code = LANGS[st.session_state.lang_label][0]
    values = item.get(field, {})
    return values.get(code) or values.get("en") or next(iter(values.values()), "")


def logout():
    st.session_state.authenticated = False
    st.session_state.ticket_no = None
    st.session_state.saved = set()
    st.session_state.viewed = []
    st.session_state.selected_id = None
    st.session_state.login_code = ""
    st.rerun()


def use_demo_ticket():
    st.session_state.login_code = "CP-ATH-0820-001"


def render_speech_button(text, key):
    """브라우저 Web Speech API를 이용한 발표용 음성 안내."""
    lang_code = LANGS[st.session_state.lang_label][1]
    safe_text = json.dumps(text)

    components.html(
        f"""
        <button onclick="speak_{key}()" style="
            width:100%;
            height:40px;
            border-radius:9px;
            border:1px solid #BBC8E6;
            background:#FFFFFF;
            color:#354F99;
            font-weight:700;
            cursor:pointer;
            font-family:Arial,sans-serif;
        ">🔊 Audio</button>

        <script>
        function speak_{key}() {{
            const text = {safe_text};
            if (!("speechSynthesis" in window)) {{
                alert("이 브라우저는 음성 합성을 지원하지 않습니다.");
                return;
            }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = "{lang_code}";
            window.speechSynthesis.speak(utterance);
        }}
        </script>
        """,
        height=45,
    )


def item_image_path(item):
    """
    heritage.json에 아래처럼 image를 추가하면 실제 사진을 표시할 수 있음.
    "image": "assets/parthenon.jpg"
    """
    image = item.get("image")
    if not image:
        return None

    p = BASE_DIR / image
    return p if p.exists() else None


# =========================================================
# 6. LOGIN PAGE
#    - "CULTURE PASS" 문구 제거
#    - 입장 버튼을 ENTER로 변경
#    - 일반 텍스트 검정색
#    - 빈 흰색 박스가 생기던 기존 div wrapper 방식 제거
# =========================================================
def login_page():
    st.markdown("<div style='height:4vh'></div>", unsafe_allow_html=True)

    outer_left, center, outer_right = st.columns([1.1, 1.35, 1.1])

    with center:
        st.markdown(
            """
            <div class="login-browser">
                <span class="dot dot-red"></span>
                <span class="dot dot-yellow"></span>
                <span class="dot dot-green"></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                """
                <div class="login-eyebrow">DIGITAL HERITAGE ACCESS</div>
                <div class="login-heading">박물관 입장권으로<br>문화유산을 다시 만나다</div>
                <div class="login-description">
                    박물관 입장권의 <b>고객번호</b>를 인증하면,
                    관람했던 문화유산의 다국어 해설·음성 안내와
                    도시 문화유산 연계 정보를 관람 이후에도 확인할 수 있습니다.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.text_input(
                "티켓 고객번호",
                key="login_code",
                placeholder="예: CP-ATH-0820-001",
            )

            c1, c2 = st.columns([1.15, 1])

            with c1:
                if st.button("ENTER", type="primary", use_container_width=True):
                    code = st.session_state.login_code.strip().upper()

                    if code in TICKETS:
                        st.session_state.authenticated = True
                        st.session_state.ticket_no = code
                        st.session_state.saved = set()
                        st.session_state.viewed = []
                        st.session_state.selected_id = None
                        st.rerun()
                    else:
                        st.error("등록되지 않은 고객번호입니다. 티켓 정보를 다시 확인해 주세요.")

            with c2:
                st.button(
                    "USE DEMO TICKET",
                    use_container_width=True,
                    on_click=use_demo_ticket,
                )

            st.markdown(
                """
                <div class="login-note">
                    <b>Demo Ticket</b> &nbsp; CP-ATH-0820-001<br>
                    발표용 프로토타입으로, 실제 서비스에서는 고객번호를
                    서버 데이터베이스에서 암호화하여 검증하는 구조로 확장할 수 있습니다.
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# 7. HEADER AFTER LOGIN
# =========================================================
def main_header(ticket):
    top_left, top_right = st.columns([4.7, 1.3], vertical_alignment="center")

    with top_left:
        st.markdown(
            "<div style='color:#5A6070;font-size:12px;font-weight:800;letter-spacing:.09em;'>"
            "DIGITAL HERITAGE GUIDE</div>",
            unsafe_allow_html=True,
        )

    with top_right:
        ctrl1, ctrl2 = st.columns([1.45, 1])
        with ctrl1:
            selected = st.selectbox(
                "Language",
                list(LANGS.keys()),
                index=list(LANGS.keys()).index(st.session_state.lang_label),
                label_visibility="collapsed",
            )
            st.session_state.lang_label = selected

        with ctrl2:
            if st.button("LOG OUT", use_container_width=True):
                logout()

    st.markdown(
        f"""
        <div class="platform-shell">
            <div class="browser-top">
                <span class="dot dot-red"></span>
                <span class="dot dot-yellow"></span>
                <span class="dot dot-green"></span>
            </div>

            <div class="platform-title-row">
                <div class="platform-brand">CULTURE PASS</div>
                <div class="route-pill">{html.escape(ticket["route"])}</div>
            </div>

            <div class="verified">
                <div class="verified-icon">🎟</div>
                <div>
                    <div class="verified-title">Ticket Verified ✓</div>
                    <div class="verified-sub">
                        {html.escape(ticket["customer_name"])}
                        &nbsp; · &nbsp;
                        방문 {html.escape(ticket["visit_date"])}
                        &nbsp; · &nbsp;
                        이용 가능 {html.escape(ticket["valid_until"])}까지
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 8. LEFT / RIGHT FUNCTION PANELS
# =========================================================
def render_core_structure():
    st.markdown(
        """
        <div class="side-panel">
            <div class="side-title">핵심 구조</div>

            <div class="side-row">
                <div class="side-row-title">01 · 티켓 기반 인증</div>
                <div class="side-row-text">
                    입장권 고객번호를 기반으로 승인된 콘텐츠에만 접근
                </div>
            </div>

            <div class="side-row">
                <div class="side-row-title">02 · 다국어 해설 모듈</div>
                <div class="side-row-text">
                    한국어·영어·그리스어 텍스트와 음성 해설 제공
                </div>
            </div>

            <div class="side-row">
                <div class="side-row-title">03 · 관람 기록 저장</div>
                <div class="side-row-text">
                    열람한 문화유산과 관심 콘텐츠를 개인 기록으로 축적
                </div>
            </div>

            <div class="side-row">
                <div class="side-row-title">04 · 통합 문화유산 연결</div>
                <div class="side-row-text">
                    박물관과 도시 공간의 문화유산 정보를 하나의 서비스에서 탐색
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_expected_features():
    st.markdown(
        """
        <div class="side-panel">
            <div class="side-title">기대 기능</div>

            <div class="side-row">
                <div class="side-row-title">◎ 접근성</div>
                <div class="side-row-text">
                    다국어 제공을 통해 정보 접근의 언어 장벽 완화
                </div>
            </div>

            <div class="side-row">
                <div class="side-row-title">▣ 지속성</div>
                <div class="side-row-text">
                    관람 종료 이후에도 해설과 학습 경험을 연장
                </div>
            </div>

            <div class="side-row">
                <div class="side-row-title">○ 개인화</div>
                <div class="side-row-text">
                    관심 문화유산 저장·회고를 통한 개인별 탐색 경험
                </div>
            </div>

            <div class="side-row">
                <div class="side-row-title">◇ 확장성</div>
                <div class="side-row-text">
                    향후 한국 박물관·문화재 플랫폼으로 서비스 구조 확장 가능
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 9. HERITAGE CARDS
# =========================================================
def heritage_cards(ticket):
    ids = [x for x in ticket["access"] if x in HERITAGE_BY_ID]
    items = [HERITAGE_BY_ID[x] for x in ids]

    if not items:
        st.info("현재 티켓으로 열람할 수 있는 문화유산이 없습니다.")
        return

    # 3개씩 나누어 배치
    for start in range(0, len(items), 3):
        row_items = items[start : start + 3]
        cols = st.columns(len(row_items))

        for col, item in zip(cols, row_items):
            with col:
                image_path = item_image_path(item)

                st.markdown(
                    f"""
                    <div class="heritage-card">
                        <div class="card-title">{html.escape(tr(item, "title"))}</div>
                        <div class="card-subtitle">{html.escape(tr(item, "subtitle"))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 카드 내부 실제 사진
                if image_path:
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.markdown(
                        f'<div class="card-visual">{item.get("emoji", "🏛️")}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f'<div class="card-meta">{html.escape(item.get("museum", ""))}</div>',
                    unsafe_allow_html=True,
                )

                a, b = st.columns(2)

                with a:
                    render_speech_button(tr(item, "description"), item["id"])

                with b:
                    if st.button(
                        "Read More",
                        key=f"read_{item['id']}",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state.selected_id = item["id"]
                        if item["id"] not in st.session_state.viewed:
                            st.session_state.viewed.append(item["id"])

                is_saved = item["id"] in st.session_state.saved

                if st.button(
                    "★ Saved" if is_saved else "☆ Save",
                    key=f"save_{item['id']}",
                    use_container_width=True,
                ):
                    if is_saved:
                        st.session_state.saved.remove(item["id"])
                    else:
                        st.session_state.saved.add(item["id"])
                    st.rerun()

            # 간격
        st.write("")


def detail_panel():
    item_id = st.session_state.selected_id

    if not item_id or item_id not in HERITAGE_BY_ID:
        return

    item = HERITAGE_BY_ID[item_id]

    st.markdown(
        f"""
        <div class="detail-box">
            <div class="detail-title">{html.escape(tr(item, "title"))}</div>
            <div class="detail-meta">
                {html.escape(item.get("museum", ""))}
                &nbsp; · &nbsp;
                {html.escape(item.get("period", ""))}
            </div>
            <div class="detail-text">
                {html.escape(tr(item, "description"))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def saved_section():
    st.markdown('<div class="section-label">My Saved Heritage</div>', unsafe_allow_html=True)

    saved_items = [
        HERITAGE_BY_ID[x]
        for x in st.session_state.saved
        if x in HERITAGE_BY_ID
    ]

    if not saved_items:
        st.caption("아직 저장된 문화유산이 없습니다.")
        return

    cols = st.columns(min(4, len(saved_items)))

    for i, item in enumerate(saved_items):
        with cols[i % len(cols)]:
            st.markdown(
                f"""
                <div class="saved-chip">
                    <div class="saved-title">
                        {item.get("emoji", "🏛️")} {html.escape(tr(item, "title"))}
                    </div>
                    <div class="saved-meta">
                        {html.escape(item.get("museum", ""))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# 10. HOME
# =========================================================
def home_page(ticket):
    left, middle, right = st.columns([1.05, 3.8, 1.05], gap="large")

    with left:
        render_core_structure()

    with middle:
        st.markdown('<div class="section-label">Heritage Guide</div>', unsafe_allow_html=True)
        heritage_cards(ticket)
        detail_panel()
        st.write("")
        saved_section()

    with right:
        render_expected_features()

    st.markdown(
        """
        <div class="quote-box">
            “정보 설계의 핵심은 데이터를 나열하는 데 있지 않고,
            사용자가 맥락 속에서 지식을 재구성하도록 지원하는 데 있다.”
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 11. MY HERITAGE
# =========================================================
def my_heritage_page(ticket):
    st.markdown('<div class="section-label">My Heritage</div>', unsafe_allow_html=True)

    a, b, c = st.columns(3)
    a.metric("Accessible Heritage", len(ticket["access"]))
    b.metric("Saved", len(st.session_state.saved))
    c.metric("Viewed", len(st.session_state.viewed))

    st.write("")
    st.markdown("#### 최근 열람 기록")

    if not st.session_state.viewed:
        st.info("상세 해설을 열람하면 이곳에 기록됩니다.")
    else:
        for item_id in reversed(st.session_state.viewed[-6:]):
            item = HERITAGE_BY_ID[item_id]
            with st.container(border=True):
                st.write(
                    f"{item.get('emoji','🏛️')} **{tr(item,'title')}**  \n"
                    f"{item.get('museum','')}"
                )

    st.write("")
    saved_section()


# =========================================================
# 12. CITY HERITAGE
# =========================================================
def city_heritage_page(ticket):
    st.markdown('<div class="section-label">City Heritage</div>', unsafe_allow_html=True)

    city_items = [
        HERITAGE_BY_ID[x]
        for x in ticket["access"]
        if x in HERITAGE_BY_ID and HERITAGE_BY_ID[x].get("type") == "city"
    ]

    st.write(
        "박물관 내부의 관람 경험을 지하철역·공공공간·고고학 유적 등 "
        "도시 일상의 문화유산 탐색으로 확장합니다."
    )

    if not city_items:
        st.info("현재 티켓에는 도시 문화유산 경로가 포함되어 있지 않습니다.")
        return

    df = pd.DataFrame(
        [
            {
                "lat": item["lat"],
                "lon": item["lon"],
                "name": tr(item, "title"),
            }
            for item in city_items
        ]
    )

    st.map(df, latitude="lat", longitude="lon", size=150)

    for item in city_items:
        with st.expander(f"{item.get('emoji','🏛️')} {tr(item,'title')}"):
            st.write(tr(item, "description"))
            st.caption(item.get("museum", ""))


# =========================================================
# 13. SEARCH
# =========================================================
def search_page(ticket):
    st.markdown('<div class="section-label">Search Heritage</div>', unsafe_allow_html=True)

    query = st.text_input(
        "문화유산 검색",
        placeholder="예: 파르테논 / Hellenic IT / Metro",
    )

    items = [
        HERITAGE_BY_ID[x]
        for x in ticket["access"]
        if x in HERITAGE_BY_ID
    ]

    if query.strip():
        q = query.strip().lower()
        result = []

        for item in items:
            searchable = " ".join(
                [
                    item.get("title", {}).get("ko", ""),
                    item.get("title", {}).get("en", ""),
                    item.get("title", {}).get("el", ""),
                    item.get("subtitle", {}).get("ko", ""),
                    item.get("subtitle", {}).get("en", ""),
                    item.get("museum", ""),
                    item.get("period", ""),
                ]
            ).lower()

            if q in searchable:
                result.append(item)
    else:
        result = items

    st.caption(f"{len(result)} results")

    for item in result:
        with st.container(border=True):
            st.markdown(f"### {item.get('emoji','🏛️')} {tr(item,'title')}")
            st.write(tr(item, "subtitle"))
            st.caption(f"{item.get('museum','')} · {item.get('period','')}")

            if st.button(
                "OPEN",
                key=f"search_open_{item['id']}",
                type="primary",
            ):
                st.session_state.selected_id = item["id"]

                if item["id"] not in st.session_state.viewed:
                    st.session_state.viewed.append(item["id"])

    detail_panel()


# =========================================================
# 14. APP
# =========================================================
def main():
    if not st.session_state.authenticated:
        login_page()
        return

    ticket = TICKETS[st.session_state.ticket_no]

    main_header(ticket)

    tabs = st.tabs(
        [
            "HOME",
            "MY HERITAGE",
            "CITY HERITAGE",
            "SEARCH",
        ]
    )

    with tabs[0]:
        home_page(ticket)

    with tabs[1]:
        my_heritage_page(ticket)

    with tabs[2]:
        city_heritage_page(ticket)

    with tabs[3]:
        search_page(ticket)


main()
