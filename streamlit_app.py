import json
import html
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Digital Heritage Guide",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# DATA
# =========================================================
@st.cache_data
def load_data():
    heritage_path = BASE_DIR / "data" / "heritage.json"
    tickets_path = BASE_DIR / "data" / "tickets.json"

    missing = [
        str(p.relative_to(BASE_DIR))
        for p in (heritage_path, tickets_path)
        if not p.exists()
    ]

    if missing:
        st.error(
            "필수 데이터 파일을 찾을 수 없습니다.\n\n"
            + "\n".join(f"- `{m}`" for m in missing)
            + "\n\nGitHub 저장소에서 `data` 폴더 안에 JSON 파일이 있는지 확인해 주세요."
        )
        st.stop()

    heritage = json.loads(heritage_path.read_text(encoding="utf-8"))
    tickets = json.loads(tickets_path.read_text(encoding="utf-8"))
    return heritage, tickets


HERITAGE, TICKETS = load_data()
HERITAGE_BY_ID = {item["id"]: item for item in HERITAGE}

LANGS = {
    "한국어": ("ko", "ko-KR"),
    "English": ("en", "en-US"),
    "Ελληνικά": ("el", "el-GR"),
}


# =========================================================
# SESSION STATE
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
# IMPORTANT: HTML RENDER HELPER
# 멀티라인 HTML의 들여쓰기가 Markdown 코드블록으로 렌더링되는 현상 방지
# =========================================================
def render_html(markup: str):
    st.markdown(
        textwrap.dedent(markup).strip(),
        unsafe_allow_html=True,
    )


# =========================================================
# STYLE
# =========================================================
render_html(
    """
    <style>
    :root {
        --page: #F7F8FC;
        --surface: #FFFFFF;
        --surface-soft: #F5F7FC;
        --blue: #4C61B8;
        --blue-deep: #354B99;
        --blue-soft: #EEF2FC;
        --line: #E0E5EF;
        --line-blue: #C8D3F0;
        --text: #171A22;
        --subtext: #555B67;
        --muted: #8B919C;
    }

    html {
        color-scheme: light !important;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--page) !important;
        color: var(--text) !important;
    }

    .block-container {
        max-width: 1540px;
        padding-top: 1.35rem;
        padding-bottom: 3rem;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* 기본 텍스트 */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    .stCaption,
    label {
        color: var(--text) !important;
    }

    /* Input */
    div[data-baseweb="input"] > div {
        background: #FFFFFF !important;
        border: 1px solid #D5DBE7 !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        min-height: 56px !important;
    }

    div[data-baseweb="input"] input {
        color: #14171D !important;
        -webkit-text-fill-color: #14171D !important;
        background: #FFFFFF !important;
        font-size: 16px !important;
        padding-left: 6px !important;
    }

    div[data-baseweb="input"] input::placeholder {
        color: #9298A4 !important;
        -webkit-text-fill-color: #9298A4 !important;
    }

    /* Select */
    div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        color: var(--text) !important;
        border: 1px solid #D9DFE9 !important;
        border-radius: 11px !important;
        min-height: 48px !important;
    }

    /* Buttons */
    div[data-testid="stButton"] button {
        min-height: 50px;
        border-radius: 11px;
        font-weight: 760;
        transition: .15s ease;
        box-shadow: none !important;
    }

    div[data-testid="stButton"] button:not([kind="primary"]) {
        background: #FFFFFF !important;
        color: #222630 !important;
        border: 1px solid #D6DDE9 !important;
    }

    div[data-testid="stButton"] button:not([kind="primary"]) p {
        color: #222630 !important;
    }

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

    /* ================= LOGIN ================= */
    .login-window-top {
        width: 100%;
        height: 48px;
        background: #FFFFFF;
        border: 1px solid #DFE3EA;
        border-bottom: 0;
        border-radius: 24px 24px 0 0;
        padding: 0 19px;
        display: flex;
        align-items: center;
        gap: 8px;
        box-sizing: border-box;
    }

    .window-dot {
        width: 11px;
        height: 11px;
        border-radius: 999px;
        display: inline-block;
    }

    .dot-red { background: #C97764; }
    .dot-yellow { background: #D5B66A; }
    .dot-green { background: #7FAF67; }

    /* login 카드가 더 넓고 여유롭게 보이도록 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border-color: #DFE3EA !important;
        border-radius: 0 0 24px 24px !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 28px 34px 30px 34px !important;
    }

    .login-eyebrow {
        color: var(--blue-deep);
        font-size: 12px;
        font-weight: 850;
        letter-spacing: .13em;
        margin-bottom: 10px;
    }

    .login-title {
        color: var(--text);
        font-family: Georgia, "Times New Roman", "Noto Serif KR", serif;
        font-size: 38px;
        line-height: 1.25;
        font-weight: 800;
        margin: 0 0 14px 0;
    }

    .login-description {
        color: #505661;
        font-size: 15px;
        line-height: 1.8;
        margin-bottom: 18px;
    }

    .login-guide {
        background: var(--blue-soft);
        border: 1px solid #DCE5FA;
        border-radius: 13px;
        color: #343A47;
        padding: 14px 16px;
        font-size: 13px;
        line-height: 1.6;
        margin-top: 10px;
    }

    /* ================= MAIN APP ================= */
    .app-kicker {
        color: #555C6D;
        font-size: 12px;
        font-weight: 850;
        letter-spacing: .10em;
        padding-top: 7px;
    }

    .platform-shell {
        background: #FFFFFF;
        border: 1px solid #DEE3EC;
        border-radius: 20px;
        box-shadow: 0 12px 34px rgba(46, 57, 82, .06);
        overflow: hidden;
        margin: 10px 0 18px 0;
    }

    .browser-top {
        height: 43px;
        display: flex;
        align-items: center;
        gap: 7px;
        padding: 0 18px;
        border-bottom: 1px solid #ECEEF3;
    }

    .platform-title-row {
        padding: 16px 22px 10px 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .platform-brand {
        color: #1D2C5B;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 22px;
        font-weight: 800;
    }

    .route-pill {
        background: #F1F4FA;
        border: 1px solid #E0E5EF;
        color: #45557D;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
    }

    .verified {
        margin: 0 22px 20px 22px;
        padding: 17px 18px;
        display: flex;
        align-items: center;
        gap: 14px;
        background: linear-gradient(90deg, #F0F4FC, #FAFBFE);
        border: 1px solid #D9E1F2;
        border-radius: 14px;
    }

    .verified-icon {
        width: 48px;
        height: 48px;
        border-radius: 999px;
        background: var(--blue);
        color: #FFFFFF;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 21px;
        flex: 0 0 auto;
    }

    .verified-title {
        color: var(--blue-deep);
        font-size: 18px;
        font-weight: 850;
    }

    .verified-sub {
        color: #414751;
        font-size: 13px;
        margin-top: 3px;
    }

    /* Side panels */
    .side-panel {
        background: #FFFFFF;
        border: 1px solid #E0E5ED;
        border-radius: 18px;
        padding: 18px 16px;
        min-height: 510px;
        box-sizing: border-box;
    }

    .side-title {
        color: var(--blue);
        text-align: center;
        font-size: 19px;
        font-weight: 850;
        padding: 2px 0 13px 0;
        border-bottom: 1px solid #DDE3EE;
    }

    .side-row {
        padding: 16px 2px;
        border-bottom: 1px solid #ECEEF3;
    }

    .side-row:last-child {
        border-bottom: none;
    }

    .side-row-title {
        color: #3652A0;
        font-size: 14px;
        font-weight: 850;
    }

    .side-row-text {
        color: #555B65;
        font-size: 12px;
        line-height: 1.6;
        margin-top: 4px;
    }

    /* Heritage */
    .section-label {
        color: #344F9F;
        font-size: 20px;
        font-weight: 850;
        margin: 5px 0 12px 0;
    }

    .heritage-card {
        background: #FFFFFF;
        border: 1px solid #E0E4EC;
        border-radius: 14px;
        padding: 15px;
        min-height: 110px;
        box-sizing: border-box;
    }

    .card-title {
        color: #425DAA;
        font-size: 18px;
        font-weight: 850;
    }

    .card-subtitle {
        color: #343942;
        font-size: 12.5px;
        line-height: 1.5;
        min-height: 40px;
        margin-top: 4px;
    }

    .card-visual {
        height: 132px;
        border-radius: 11px;
        border: 1px solid #E3E7EE;
        background: linear-gradient(135deg, #EEF2FA, #F8F6F2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 64px;
        margin: 9px 0 10px 0;
    }

    .card-meta {
        color: #808590;
        font-size: 11px;
        min-height: 28px;
        margin: 4px 0 6px 0;
    }

    .detail-box {
        background: #FFFFFF;
        border: 1px solid #DCE3F2;
        border-radius: 15px;
        padding: 18px;
        margin-top: 15px;
    }

    .detail-title {
        color: #314D99;
        font-size: 21px;
        font-weight: 850;
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
        padding: 10px 11px;
        min-height: 67px;
    }

    .saved-title {
        color: #20242D;
        font-size: 13px;
        font-weight: 760;
    }

    .saved-meta {
        color: #898F99;
        font-size: 10.5px;
        margin-top: 3px;
    }

    .quote-box {
        background: #EEF3FC;
        border: 1px solid #DBE4F8;
        border-radius: 14px;
        color: #405BA5;
        padding: 16px 20px;
        text-align: center;
        font-size: 15px;
        font-weight: 750;
        margin-top: 17px;
    }

    /* Tabs */
    div[data-baseweb="tab-list"] {
        gap: 8px;
    }

    button[data-baseweb="tab"] {
        color: #555C68 !important;
        font-weight: 750 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #344F9F !important;
    }

    @media (max-width: 900px) {
        .login-title { font-size: 30px; }
        .side-panel { min-height: auto; }
    }
    </style>
    """
)


# =========================================================
# HELPERS
# =========================================================
def tr(item, field):
    lang_code = LANGS[st.session_state.lang_label][0]
    values = item.get(field, {})
    return values.get(lang_code) or values.get("en") or next(iter(values.values()), "")


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


def image_path(item):
    value = item.get("image")
    if not value:
        return None

    p = BASE_DIR / value
    return p if p.exists() else None


def speech_button(text, key):
    safe_text = json.dumps(text)
    speech_lang = LANGS[st.session_state.lang_label][1]

    components.html(
        f"""
        <button
          onclick="speak_{key}()"
          style="
            width:100%;
            height:42px;
            border-radius:9px;
            border:1px solid #BAC7E4;
            background:#FFFFFF;
            color:#354F99;
            font-weight:700;
            cursor:pointer;
            font-family:Arial,sans-serif;
          "
        >
          🔊 Audio
        </button>

        <script>
        function speak_{key}() {{
            if (!("speechSynthesis" in window)) {{
                alert("이 브라우저는 음성 합성을 지원하지 않습니다.");
                return;
            }}

            window.speechSynthesis.cancel();

            const utterance = new SpeechSynthesisUtterance({safe_text});
            utterance.lang = "{speech_lang}";
            window.speechSynthesis.speak(utterance);
        }}
        </script>
        """,
        height=47,
    )


# =========================================================
# LOGIN
# =========================================================
def login_page():
    st.markdown("<div style='height:5vh'></div>", unsafe_allow_html=True)

    # 기존보다 카드 폭 확대
    left_space, center, right_space = st.columns([0.7, 1.85, 0.7])

    with center:
        render_html(
            """
            <div class="login-window-top">
                <span class="window-dot dot-red"></span>
                <span class="window-dot dot-yellow"></span>
                <span class="window-dot dot-green"></span>
            </div>
            """
        )

        with st.container(border=True):
            render_html(
                """
                <div class="login-eyebrow">DIGITAL HERITAGE ACCESS</div>
                <div class="login-title">
                    박물관 입장권으로<br>
                    문화유산을 다시 만나다
                </div>
                <div class="login-description">
                    박물관 입장권의 <b>고객번호</b>를 인증하면,
                    관람했던 문화유산의 다국어 해설과 음성 안내,
                    그리고 도시 문화유산 연계 정보를 관람 이후에도 다시 확인할 수 있습니다.
                </div>
                """
            )

            st.text_input(
                "티켓 고객번호",
                key="login_code",
                placeholder="예: CP-ATH-0820-001",
            )

            btn1, btn2 = st.columns([1.1, 1])

            with btn1:
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

            with btn2:
                st.button(
                    "USE DEMO TICKET",
                    use_container_width=True,
                    on_click=use_demo_ticket,
                )

            render_html(
                """
                <div class="login-guide">
                    <b>Demo Ticket</b> &nbsp; CP-ATH-0820-001<br>
                    발표용 프로토타입입니다. 실제 서비스에서는 고객번호를
                    서버 데이터베이스에서 암호화하여 검증하는 구조로 확장할 수 있습니다.
                </div>
                """
            )


# =========================================================
# HEADER
# =========================================================
def main_header(ticket):
    left, right = st.columns([4.8, 1.2], vertical_alignment="center")

    with left:
        render_html(
            '<div class="app-kicker">DIGITAL HERITAGE GUIDE</div>'
        )

    with right:
        a, b = st.columns([1.45, 1])

        with a:
            selected_lang = st.selectbox(
                "Language",
                list(LANGS.keys()),
                index=list(LANGS.keys()).index(st.session_state.lang_label),
                label_visibility="collapsed",
            )
            st.session_state.lang_label = selected_lang

        with b:
            if st.button("LOG OUT", use_container_width=True):
                logout()

    route = html.escape(ticket.get("route", ""))
    name = html.escape(ticket.get("customer_name", ""))
    visit = html.escape(ticket.get("visit_date", ""))
    valid = html.escape(ticket.get("valid_until", ""))

    render_html(
        f"""
        <div class="platform-shell">
            <div class="browser-top">
                <span class="window-dot dot-red"></span>
                <span class="window-dot dot-yellow"></span>
                <span class="window-dot dot-green"></span>
            </div>

            <div class="platform-title-row">
                <div class="platform-brand">CULTURE PASS</div>
                <div class="route-pill">{route}</div>
            </div>

            <div class="verified">
                <div class="verified-icon">🎟</div>
                <div>
                    <div class="verified-title">Ticket Verified ✓</div>
                    <div class="verified-sub">
                        {name} · 방문 {visit} · 이용 가능 {valid}까지
                    </div>
                </div>
            </div>
        </div>
        """
    )


# =========================================================
# SIDE PANELS
# =========================================================
def core_structure():
    render_html(
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
                    박물관과 도시 공간의 문화유산을 하나의 탐색 경험으로 연결
                </div>
            </div>
        </div>
        """
    )


def expected_features():
    render_html(
        """
        <div class="side-panel">
            <div class="side-title">기대 기능</div>

            <div class="side-row">
                <div class="side-row-title">◎ 접근성</div>
                <div class="side-row-text">
                    다국어 해설을 통해 정보 접근의 언어 장벽을 완화
                </div>
            </div>

            <div class="side-row">
                <div class="side-row-title">▣ 지속성</div>
                <div class="side-row-text">
                    관람이 끝난 뒤에도 해설과 학습 경험을 지속
                </div>
            </div>

            <div class="side-row">
                <div class="side-row-title">○ 개인화</div>
                <div class="side-row-text">
                    관심 문화유산 저장과 회고를 통한 개인별 탐색 경험
                </div>
            </div>

            <div class="side-row">
                <div class="side-row-title">◇ 확장성</div>
                <div class="side-row-text">
                    향후 한국의 박물관·문화유산 서비스로 구조 확장 가능
                </div>
            </div>
        </div>
        """
    )


# =========================================================
# HERITAGE CARDS
# =========================================================
def heritage_cards(ticket):
    items = [
        HERITAGE_BY_ID[item_id]
        for item_id in ticket.get("access", [])
        if item_id in HERITAGE_BY_ID
    ]

    if not items:
        st.info("현재 티켓으로 열람 가능한 문화유산이 없습니다.")
        return

    for start in range(0, len(items), 3):
        row = items[start:start + 3]
        cols = st.columns(len(row))

        for col, item in zip(cols, row):
            with col:
                render_html(
                    f"""
                    <div class="heritage-card">
                        <div class="card-title">{html.escape(tr(item, "title"))}</div>
                        <div class="card-subtitle">{html.escape(tr(item, "subtitle"))}</div>
                    </div>
                    """
                )

                p = image_path(item)

                if p:
                    st.image(str(p), use_container_width=True)
                else:
                    render_html(
                        f'<div class="card-visual">{item.get("emoji", "🏛️")}</div>'
                    )

                render_html(
                    f'<div class="card-meta">{html.escape(item.get("museum", ""))}</div>'
                )

                c1, c2 = st.columns(2)

                with c1:
                    speech_button(tr(item, "description"), item["id"])

                with c2:
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

        st.write("")


def detail_panel():
    item_id = st.session_state.selected_id

    if not item_id or item_id not in HERITAGE_BY_ID:
        return

    item = HERITAGE_BY_ID[item_id]

    render_html(
        f"""
        <div class="detail-box">
            <div class="detail-title">{html.escape(tr(item, "title"))}</div>
            <div class="detail-meta">
                {html.escape(item.get("museum", ""))}
                ·
                {html.escape(item.get("period", ""))}
            </div>
            <div class="detail-text">
                {html.escape(tr(item, "description"))}
            </div>
        </div>
        """
    )


def saved_section():
    render_html('<div class="section-label">My Saved Heritage</div>')

    saved_items = [
        HERITAGE_BY_ID[item_id]
        for item_id in st.session_state.saved
        if item_id in HERITAGE_BY_ID
    ]

    if not saved_items:
        st.caption("아직 저장된 문화유산이 없습니다.")
        return

    cols = st.columns(min(4, len(saved_items)))

    for idx, item in enumerate(saved_items):
        with cols[idx % len(cols)]:
            render_html(
                f"""
                <div class="saved-chip">
                    <div class="saved-title">
                        {item.get("emoji", "🏛️")} {html.escape(tr(item, "title"))}
                    </div>
                    <div class="saved-meta">
                        {html.escape(item.get("museum", ""))}
                    </div>
                </div>
                """
            )


# =========================================================
# HOME
# =========================================================
def home_page(ticket):
    left, middle, right = st.columns([1.05, 3.8, 1.05], gap="large")

    with left:
        core_structure()

    with middle:
        render_html('<div class="section-label">Heritage Guide</div>')
        heritage_cards(ticket)
        detail_panel()
        st.write("")
        saved_section()

    with right:
        expected_features()

    render_html(
        """
        <div class="quote-box">
            “정보 설계의 핵심은 데이터를 나열하는 데 있지 않고,
            사용자가 맥락 속에서 지식을 재구성하도록 지원하는 데 있다.”
        </div>
        """
    )


# =========================================================
# MY HERITAGE
# =========================================================
def my_heritage_page(ticket):
    render_html('<div class="section-label">My Heritage</div>')

    a, b, c = st.columns(3)

    a.metric("Accessible Heritage", len(ticket.get("access", [])))
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
                    f"{item.get('emoji', '🏛️')} **{tr(item, 'title')}**  \n"
                    f"{item.get('museum', '')}"
                )

    st.write("")
    saved_section()


# =========================================================
# CITY HERITAGE
# =========================================================
def city_heritage_page(ticket):
    render_html('<div class="section-label">City Heritage</div>')

    city_items = [
        HERITAGE_BY_ID[item_id]
        for item_id in ticket.get("access", [])
        if item_id in HERITAGE_BY_ID
        and HERITAGE_BY_ID[item_id].get("type") == "city"
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
        with st.expander(f"{item.get('emoji', '🏛️')} {tr(item, 'title')}"):
            st.write(tr(item, "description"))
            st.caption(item.get("museum", ""))


# =========================================================
# SEARCH
# =========================================================
def search_page(ticket):
    render_html('<div class="section-label">Search Heritage</div>')

    query = st.text_input(
        "문화유산 검색",
        placeholder="예: 파르테논 / Hellenic IT / Metro",
    )

    items = [
        HERITAGE_BY_ID[item_id]
        for item_id in ticket.get("access", [])
        if item_id in HERITAGE_BY_ID
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
            st.markdown(f"### {item.get('emoji', '🏛️')} {tr(item, 'title')}")
            st.write(tr(item, "subtitle"))
            st.caption(f"{item.get('museum', '')} · {item.get('period', '')}")

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
# APP
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
