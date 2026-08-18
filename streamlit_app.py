
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CULTURE PASS",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# DATA
# =========================================================
def load_data():
    heritage_path = BASE_DIR / "data" / "heritage.json"
    tickets_path = BASE_DIR / "data" / "tickets.json"

    if not heritage_path.exists() or not tickets_path.exists():
        st.error(
            "data 폴더 안에 heritage.json과 tickets.json이 필요합니다.\n\n"
            "구조 예시:\n"
            "data/heritage.json\n"
            "data/tickets.json"
        )
        st.stop()

    heritage = json.loads(heritage_path.read_text(encoding="utf-8"))
    tickets = json.loads(tickets_path.read_text(encoding="utf-8"))
    return heritage, tickets


HERITAGE, TICKETS = load_data()

# ------------------------------------------------------------------
# Runtime data normalization
# - Kerameikos is removed everywhere.
# - The old "metro_heritage" slot is repurposed as "Acropolis Museum".
#   Keeping the internal id "metro_heritage" means existing tickets.json
#   files continue to work without requiring another GitHub edit.
# ------------------------------------------------------------------
ACROPOLIS_MUSEUM_ITEM = {
    "id": "metro_heritage",
    "type": "museum",
    "title": {
        "ko": "아크로폴리스 박물관",
        "en": "Acropolis Museum",
        "el": "Μουσείο Ακρόπολης"
    },
    "subtitle": {
        "ko": "아크로폴리스 유물과 디지털 해설을 연결하는 박물관",
        "en": "Connecting Acropolis collections with digital interpretation",
        "el": "Σύνδεση των συλλογών της Ακρόπολης με την ψηφιακή ερμηνεία"
    },
    "description": {
        "ko": (
            "아크로폴리스 박물관은 아테네 아크로폴리스에서 출토되거나 "
            "그 유적과 직접 관련된 문화유산을 중심으로 전시합니다. "
            "공식 안내에 따르면 박물관 1층 서쪽에는 프로필라이아, "
            "아테나 니케 신전, 에레크테이온의 조각과 건축 부재가 전시되며 "
            "카리아티드도 주요 전시품에 포함됩니다. 또한 박물관은 Smartify와 "
            "협력한 무료 디지털 가이드를 제공하여 개인 스마트기기에서 "
            "상설 컬렉션을 인터랙티브하게 탐색할 수 있도록 지원합니다."
        ),
        "en": (
            "The Acropolis Museum presents cultural heritage associated directly "
            "with the Athenian Acropolis. According to the Museum, the west side "
            "of the first floor displays sculptures and architectural members "
            "from the Propylaia, the Temple of Athena Nike and the Erechtheion, "
            "with the Caryatids among its major highlights. The Museum also offers "
            "a free digital guide developed with Smartify, allowing visitors to "
            "explore the permanent collections interactively on personal smart devices."
        ),
        "el": (
            "Το Μουσείο Ακρόπολης παρουσιάζει πολιτιστικά τεκμήρια που συνδέονται "
            "άμεσα με την Αθηναϊκή Ακρόπολη. Στη δυτική πλευρά του πρώτου ορόφου "
            "εκτίθενται γλυπτά και αρχιτεκτονικά μέλη από τα Προπύλαια, τον Ναό "
            "της Αθηνάς Νίκης και το Ερέχθειο. Το Μουσείο προσφέρει επίσης δωρεάν "
            "ψηφιακό οδηγό σε συνεργασία με το Smartify για διαδραστική εξερεύνηση "
            "των μόνιμων συλλογών μέσω προσωπικών έξυπνων συσκευών."
        )
    },
    "museum": "Acropolis Museum",
    "period": "Archaic to Classical Greece",
    "lat": 37.9684,
    "lon": 23.7285,
    "emoji": "🏛️",
    "image": "assets/acropolis_museum.jpg"
}

normalized_heritage = []
acropolis_added = False

for item in HERITAGE:
    item_id = item.get("id")

    # Remove Kerameikos completely
    if item_id == "kerameikos":
        continue

    # Replace either the old metro item or a previously-created acropolis item
    if item_id in {"metro_heritage", "acropolis_museum"}:
        if not acropolis_added:
            normalized_item = ACROPOLIS_MUSEUM_ITEM.copy()
            # If heritage.json already has a custom image path, keep it.
            if item.get("image"):
                normalized_item["image"] = item["image"]
            normalized_heritage.append(normalized_item)
            acropolis_added = True
        continue

    normalized_heritage.append(item)

# If the old JSON happens not to contain the third item, still add it.
if not acropolis_added:
    normalized_heritage.append(ACROPOLIS_MUSEUM_ITEM.copy())

HERITAGE = normalized_heritage

# Normalize ticket permissions too.
for ticket in TICKETS.values():
    access = []
    for item_id in ticket.get("access", []):
        if item_id == "kerameikos":
            continue
        if item_id == "acropolis_museum":
            item_id = "metro_heritage"
        if item_id not in access:
            access.append(item_id)

    # Athens-oriented tickets that previously contained the metro slot
    # automatically continue to include the Acropolis Museum.
    ticket["access"] = access

HERITAGE_BY_ID = {item["id"]: item for item in HERITAGE}

LANGS = {
    "한국어": ("ko", "ko-KR"),
    "English": ("en", "en-US"),
    "Ελληνικά": ("el", "el-GR"),
}

# =========================================================
# GLOBAL UI TRANSLATIONS
# 로그인 첫 페이지는 이전 요청대로 영어 고정.
# 로그인 이후에는 우측 상단 언어 선택에 따라
# 화면의 모든 UI 문구가 함께 바뀝니다.
# =========================================================
UI_TEXT = {
    "ko": {
        "logout": "로그아웃",
        "ticket_verified": "티켓 인증 완료",
        "visit": "방문",
        "available_until": "이용 가능",
        "until_suffix": "까지",
        "home": "홈",
        "my_heritage_tab": "나의 문화유산",
        "city_heritage_tab": "도시 문화유산",
        "search_tab": "검색",
        "heritage_guide": "문화유산 가이드",
        "audio": "음성 해설",
        "read_more": "자세히 보기",
        "save": "저장",
        "saved": "저장됨",
        "saved_count": "저장",
        "my_saved_heritage": "저장한 문화유산",
        "no_saved": "아직 저장한 문화유산이 없습니다.",
        "my_heritage": "나의 문화유산",
        "accessible": "열람 가능",
        "viewed": "열람 기록",
        "recent_views": "최근 열람 기록",
        "no_recent_views": "‘자세히 보기’를 누르면 이곳에 열람 기록이 남습니다.",
        "city_heritage": "도시 문화유산",
        "no_city": "현재 티켓에 연결된 도시 문화유산이 없습니다.",
        "search_heritage": "문화유산 검색",
        "search_label": "문화유산 검색",
        "search_placeholder": "예: 파르테논 / 헬레닉 IT 뮤지엄 / 아크로폴리스 박물관",
        "results": "개 결과",
        "open": "열기",
        "museum": "박물관",
        "period": "시대",
    },
    "en": {
        "logout": "LOG OUT",
        "ticket_verified": "Ticket Verified",
        "visit": "Visit",
        "available_until": "Available until",
        "until_suffix": "",
        "home": "HOME",
        "my_heritage_tab": "MY HERITAGE",
        "city_heritage_tab": "CITY HERITAGE",
        "search_tab": "SEARCH",
        "heritage_guide": "Heritage Guide",
        "audio": "Audio",
        "read_more": "Read More",
        "save": "Save",
        "saved": "Saved",
        "saved_count": "Saved",
        "my_saved_heritage": "My Saved Heritage",
        "no_saved": "No heritage items have been saved yet.",
        "my_heritage": "My Heritage",
        "accessible": "Accessible",
        "viewed": "Viewed",
        "recent_views": "Recent Views",
        "no_recent_views": "Items opened with ‘Read More’ will appear here.",
        "city_heritage": "City Heritage",
        "no_city": "No city heritage locations are linked to this ticket.",
        "search_heritage": "Search Heritage",
        "search_label": "Search heritage",
        "search_placeholder": "e.g. Parthenon / Hellenic IT Museum / Acropolis Museum",
        "results": " results",
        "open": "OPEN",
        "museum": "Museum",
        "period": "Period",
    },
    "el": {
        "logout": "ΑΠΟΣΥΝΔΕΣΗ",
        "ticket_verified": "Το εισιτήριο επαληθεύτηκε",
        "visit": "Επίσκεψη",
        "available_until": "Πρόσβαση έως",
        "until_suffix": "",
        "home": "ΑΡΧΙΚΗ",
        "my_heritage_tab": "Η ΚΛΗΡΟΝΟΜΙΑ ΜΟΥ",
        "city_heritage_tab": "ΚΛΗΡΟΝΟΜΙΑ ΤΗΣ ΠΟΛΗΣ",
        "search_tab": "ΑΝΑΖΗΤΗΣΗ",
        "heritage_guide": "Οδηγός πολιτιστικής κληρονομιάς",
        "audio": "Ηχητική ξενάγηση",
        "read_more": "Περισσότερα",
        "save": "Αποθήκευση",
        "saved": "Αποθηκεύτηκε",
        "saved_count": "Αποθηκευμένα",
        "my_saved_heritage": "Αποθηκευμένη κληρονομιά",
        "no_saved": "Δεν έχουν αποθηκευτεί ακόμη στοιχεία πολιτιστικής κληρονομιάς.",
        "my_heritage": "Η κληρονομιά μου",
        "accessible": "Διαθέσιμα",
        "viewed": "Προβληθέντα",
        "recent_views": "Πρόσφατες προβολές",
        "no_recent_views": "Τα στοιχεία που ανοίγετε με «Περισσότερα» θα εμφανίζονται εδώ.",
        "city_heritage": "Πολιτιστική κληρονομιά της πόλης",
        "no_city": "Δεν υπάρχουν σημεία αστικής πολιτιστικής κληρονομιάς συνδεδεμένα με αυτό το εισιτήριο.",
        "search_heritage": "Αναζήτηση πολιτιστικής κληρονομιάς",
        "search_label": "Αναζήτηση πολιτιστικής κληρονομιάς",
        "search_placeholder": "π.χ. Παρθενώνας / Ελληνικό Μουσείο Πληροφορικής / Μουσείο Ακρόπολης",
        "results": " αποτελέσματα",
        "open": "ΑΝΟΙΓΜΑ",
        "museum": "Μουσείο",
        "period": "Περίοδος",
    },
}

# 티켓에 들어 있는 영문 데이터도 선택한 언어에 맞게 표시
TICKET_TEXT = {
    "customer_name": {
        "Demo Visitor": {
            "ko": "데모 방문객",
            "en": "Demo Visitor",
            "el": "Επισκέπτης επίδειξης",
        },
        "Museum Guest": {
            "ko": "박물관 방문객",
            "en": "Museum Guest",
            "el": "Επισκέπτης μουσείου",
        },
        "Heritage Guest": {
            "ko": "문화유산 방문객",
            "en": "Heritage Guest",
            "el": "Επισκέπτης πολιτιστικής κληρονομιάς",
        },
    },
    "route": {
        "Athens Culture Route": {
            "ko": "아테네 문화 탐방 경로",
            "en": "Athens Culture Route",
            "el": "Πολιτιστική διαδρομή Αθήνας",
        },
        "Hellenic IT Museum": {
            "ko": "헬레닉 IT 뮤지엄",
            "en": "Hellenic IT Museum",
            "el": "Ελληνικό Μουσείο Πληροφορικής",
        },
        "Athens Heritage Route": {
            "ko": "아테네 문화유산 탐방 경로",
            "en": "Athens Heritage Route",
            "el": "Διαδρομή πολιτιστικής κληρονομιάς Αθήνας",
        },
    },
}

# heritage.json에서 아직 영문 문자열로 저장된 박물관명/시대 정보 번역
HERITAGE_META = {
    "parthenon": {
        "museum": {
            "ko": "아크로폴리스 박물관 / 아크로폴리스 관람 경로",
            "en": "Acropolis Museum / Acropolis Route",
            "el": "Μουσείο Ακρόπολης / Διαδρομή Ακρόπολης",
        },
        "period": {
            "ko": "고전기 그리스",
            "en": "Classical Greece",
            "el": "Κλασική Ελλάδα",
        },
    },
    "hellenic_it": {
        "museum": {
            "ko": "헬레닉 IT 뮤지엄",
            "en": "Hellenic IT Museum",
            "el": "Ελληνικό Μουσείο Πληροφορικής",
        },
        "period": {
            "ko": "근현대 기술 유산",
            "en": "Modern Technology Heritage",
            "el": "Νεότερη τεχνολογική κληρονομιά",
        },
    },
    "metro_heritage": {
        "museum": {
            "ko": "아크로폴리스 박물관",
            "en": "Acropolis Museum",
            "el": "Μουσείο Ακρόπολης",
        },
        "period": {
            "ko": "고졸기~고전기 그리스",
            "en": "Archaic to Classical Greece",
            "el": "Αρχαϊκή έως Κλασική Ελλάδα",
        },
    },
}


# =========================================================
# SESSION
# =========================================================
DEFAULTS = {
    "authenticated": False,
    "ticket_no": None,
    "saved": set(),
    "viewed": [],
    "selected_id": None,
    "lang_label": "한국어",
    "login_code": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    :root{
        --page:#F7F8FC;
        --surface:#FFFFFF;
        --blue:#4C61B8;
        --blue-deep:#354B99;
        --blue-soft:#EEF2FC;
        --line:#E0E5EF;
        --text:#171A22;
        --sub:#5B616C;
        --muted:#8A909A;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp{
        background:var(--page)!important;
        color:var(--text)!important;
    }

    .block-container{
        max-width:1420px;
        padding-top:1.5rem;
        padding-bottom:3rem;
    }

    #MainMenu, footer, header{visibility:hidden;}

    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    .stCaption, label{
        color:var(--text)!important;
    }

    /* 입력창 */
    div[data-baseweb="input"] > div{
        background:#FFFFFF!important;
        border:1px solid #D7DDE8!important;
        border-radius:12px!important;
        min-height:56px!important;
        box-shadow:none!important;
    }

    div[data-baseweb="input"] input{
        background:#FFFFFF!important;
        color:#171A22!important;
        -webkit-text-fill-color:#171A22!important;
        font-size:16px!important;
    }

    div[data-baseweb="input"] input::placeholder{
        color:#969CA7!important;
        -webkit-text-fill-color:#969CA7!important;
    }

    /* 버튼 */
    div[data-testid="stButton"] button{
        border-radius:11px!important;
        min-height:48px;
        font-weight:750;
    }

    div[data-testid="stButton"] button[kind="primary"]{
        background:var(--blue)!important;
        color:white!important;
        border:1px solid var(--blue)!important;
    }

    div[data-testid="stButton"] button[kind="primary"] p{
        color:white!important;
    }

    div[data-testid="stButton"] button:not([kind="primary"]){
        background:white!important;
        color:#232731!important;
        border:1px solid #D5DBE7!important;
    }

    div[data-testid="stButton"] button:not([kind="primary"]) p{
        color:#232731!important;
    }

    /* select */
    div[data-baseweb="select"] > div{
        background:white!important;
        color:#171A22!important;
        border:1px solid #D8DEE9!important;
        border-radius:11px!important;
        min-height:48px!important;
    }

    /* 로그인 전체 카드: 내부 여백을 늘려 하단 Demo 박스와 외곽선이 겹치지 않게 */
    div[data-testid="stVerticalBlockBorderWrapper"]{
        border-radius:20px !important;
        overflow:visible !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div{
        padding:30px 32px 34px 32px !important;
    }

    /* 로그인 카드 */
    .login-title{
        font-family:Georgia,"Times New Roman","Noto Serif KR",serif;
        font-size:40px;
        font-weight:800;
        line-height:1.22;
        color:#171A22;
        margin-bottom:12px;
    }

    .login-kicker{
        color:var(--blue-deep);
        font-weight:850;
        font-size:12px;
        letter-spacing:.13em;
        margin-bottom:10px;
    }

    .login-desc{
        color:#555B65;
        line-height:1.75;
        font-size:15px;
        margin-bottom:18px;
    }

    .demo-box{
        background:var(--blue-soft);
        border:1px solid #DDE5F8;
        border-radius:12px;
        padding:12px 16px;
        color:#3B414C;
        font-size:13px;
        line-height:1.5;
        margin:14px 0 8px 0;
        box-sizing:border-box;
    }

    /* 상단 */
    .brand{
        font-family:Georgia,"Times New Roman",serif;
        font-weight:800;
        font-size:24px;
        color:#1E2F61;
        letter-spacing:.02em;
    }

    .ticket-box{
        background:linear-gradient(90deg,#F0F4FC 0%,#FBFCFE 100%);
        border:1px solid #D9E2F3;
        border-radius:15px;
        padding:18px 20px;
        margin:10px 0 18px 0;
    }

    .ticket-title{
        color:var(--blue-deep);
        font-weight:850;
        font-size:20px;
        margin-bottom:4px;
    }

    .ticket-meta{
        color:#454B56;
        font-size:13px;
        line-height:1.55;
    }

    /* 카드 */
    .section-title{
        color:#334E9D;
        font-size:20px;
        font-weight:850;
        margin:8px 0 12px 0;
    }

    .card-head{
        background:#FFFFFF;
        border:1px solid #E0E4EC;
        border-radius:14px 14px 0 0;
        padding:15px 15px 10px 15px;
        min-height:92px;
    }

    .card-title{
        color:#425DAB;
        font-size:18px;
        font-weight:850;
        margin-bottom:4px;
    }

    .card-sub{
        color:#3D424C;
        font-size:12.5px;
        line-height:1.45;
    }

    .card-visual{
        background:linear-gradient(135deg,#EDF2FB,#F8F6F2);
        border:1px solid #E3E7EE;
        border-top:0;
        border-radius:0 0 14px 14px;
        height:170px;
        display:flex;
        justify-content:center;
        align-items:center;
        font-size:72px;
        margin-bottom:8px;
    }

    .card-meta{
        color:#858B95;
        font-size:11px;
        min-height:27px;
        margin-bottom:6px;
    }

    .detail-box{
        background:white;
        border:1px solid #DCE3F1;
        border-radius:14px;
        padding:18px;
        margin-top:15px;
    }

    .detail-title{
        color:#314C98;
        font-size:21px;
        font-weight:850;
    }

    .detail-meta{
        color:#858B95;
        font-size:12px;
        margin:3px 0 10px 0;
    }

    .detail-text{
        color:#272B33;
        font-size:14px;
        line-height:1.75;
    }

    .saved-card{
        background:white;
        border:1px solid #E0E4EC;
        border-radius:11px;
        padding:10px 12px;
        min-height:68px;
    }

    .saved-title{
        color:#20242C;
        font-size:13px;
        font-weight:750;
    }

    .saved-meta{
        color:#8B919B;
        font-size:10.5px;
        margin-top:3px;
    }

    div[data-baseweb="tab-list"]{
        gap:8px;
    }

    button[data-baseweb="tab"]{
        color:#565D68!important;
        font-weight:750!important;
    }

    button[data-baseweb="tab"][aria-selected="true"]{
        color:#344F9F!important;
    }

    @media(max-width:900px){
        .login-title{font-size:31px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def current_lang():
    return LANGS[st.session_state.lang_label][0]


def ui(key):
    lang = current_lang()
    return UI_TEXT.get(lang, UI_TEXT["en"]).get(key, UI_TEXT["en"].get(key, key))


def tr(item, field):
    lang = current_lang()
    values = item.get(field, {})
    if isinstance(values, dict):
        return values.get(lang) or values.get("en") or next(iter(values.values()), "")
    return str(values)


def ticket_tr(ticket, field):
    lang = current_lang()
    raw = ticket.get(field, "")
    translated = TICKET_TEXT.get(field, {}).get(raw, {})
    return translated.get(lang) or translated.get("en") or raw


def meta_tr(item, field):
    lang = current_lang()

    # 명시적 다국어 메타데이터가 JSON에 들어 있다면 우선 사용
    raw = item.get(field, "")
    if isinstance(raw, dict):
        return raw.get(lang) or raw.get("en") or next(iter(raw.values()), "")

    # 현재 프로젝트 데이터의 영문 문자열은 ID별 번역표로 변환
    translated = HERITAGE_META.get(item.get("id"), {}).get(field, {})
    return translated.get(lang) or translated.get("en") or str(raw)


def logout():
    st.session_state.authenticated = False
    st.session_state.ticket_no = None
    st.session_state.saved = set()
    st.session_state.viewed = []
    st.session_state.selected_id = None
    st.session_state.login_code = ""
    st.rerun()


def use_demo():
    st.session_state.login_code = "CP-ATH-0820-001"


def actual_image(item):
    image = item.get("image")
    if not image:
        return None
    path = BASE_DIR / image
    return path if path.exists() else None


def audio_button(text, key):
    safe = json.dumps(text)
    lang_code = LANGS[st.session_state.lang_label][1]

    components.html(
        f"""
        <button onclick="speak_{key}()" style="
            width:100%;
            height:42px;
            border-radius:9px;
            border:1px solid #BAC7E4;
            background:#FFFFFF;
            color:#354F99;
            font-weight:700;
            cursor:pointer;
        ">🔊 {ui("audio")}</button>
        <script>
        function speak_{key}(){{
            if (!("speechSynthesis" in window)) return;
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance({safe});
            u.lang = "{lang_code}";
            window.speechSynthesis.speak(u);
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
    left, center, right = st.columns([0.9, 1.7, 0.9])

    with center:
        with st.container(border=True):
            st.markdown(
                '<div class="login-kicker">DIGITAL HERITAGE ACCESS</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="login-title">Your Museum Ticket,<br>Your Heritage Journey Continues</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="login-desc">Enter the <b>customer number</b> on your museum ticket '
                'to revisit multilingual heritage guides and audio commentary even after your visit.</div>',
                unsafe_allow_html=True,
            )

            st.text_input(
                "Ticket Customer Number",
                key="login_code",
                placeholder="Example: CP-ATH-0820-001",
            )

            c1, c2 = st.columns([1.1, 1])

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
                        st.error("This customer number is not registered. Please check your ticket and try again.")

            with c2:
                st.button(
                    "USE DEMO TICKET",
                    on_click=use_demo,
                    use_container_width=True,
                )

            st.markdown(
                '<div class="demo-box"><b>Demo Ticket</b> &nbsp; CP-ATH-0820-001</div>',
                unsafe_allow_html=True,
            )
            # 외곽 컨테이너 하단과 Demo Ticket 박스 사이의 안전 여백
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================
def header(ticket):
    left, right = st.columns([4.7, 1.3], vertical_alignment="center")

    with left:
        # 서비스명은 고유 브랜드명으로 유지
        st.markdown('<div class="brand">DIGITAL HERITAGE ACCESS</div>', unsafe_allow_html=True)

    with right:
        a, b = st.columns([1.45, 1])
        with a:
            new_lang = st.selectbox(
                "Language",
                list(LANGS.keys()),
                index=list(LANGS.keys()).index(st.session_state.lang_label),
                label_visibility="collapsed",
            )
            st.session_state.lang_label = new_lang

        with b:
            if st.button(ui("logout"), use_container_width=True):
                logout()

    valid_suffix = ui("until_suffix")
    st.markdown(
        f"""
        <div class="ticket-box">
            <div class="ticket-title">🎟 {ui("ticket_verified")} ✓</div>
            <div class="ticket-meta">
                {ticket_tr(ticket, "customer_name")} ·
                {ticket_tr(ticket, "route")} ·
                {ui("visit")} {ticket.get("visit_date","")} ·
                {ui("available_until")} {ticket.get("valid_until","")}{valid_suffix}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# CARDS
# =========================================================
def heritage_cards(ticket):
    items = [
        HERITAGE_BY_ID[item_id]
        for item_id in ticket.get("access", [])
        if item_id in HERITAGE_BY_ID
    ]

    for start in range(0, len(items), 3):
        row = items[start:start + 3]
        cols = st.columns(len(row), gap="medium")

        for col, item in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="card-head">
                        <div class="card-title">{tr(item,"title")}</div>
                        <div class="card-sub">{tr(item,"subtitle")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                img = actual_image(item)

                if img:
                    st.image(str(img), use_container_width=True)
                else:
                    st.markdown(
                        f'<div class="card-visual">{item.get("emoji","🏛️")}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f'<div class="card-meta">{meta_tr(item, "museum")}</div>',
                    unsafe_allow_html=True,
                )

                b1, b2 = st.columns(2)

                with b1:
                    audio_button(tr(item, "description"), item["id"])

                with b2:
                    if st.button(
                        ui("read_more"),
                        key=f"read_{item['id']}",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state.selected_id = item["id"]
                        if item["id"] not in st.session_state.viewed:
                            st.session_state.viewed.append(item["id"])

                saved = item["id"] in st.session_state.saved

                if st.button(
                    f"★ {ui('saved')}" if saved else f"☆ {ui('save')}",
                    key=f"save_{item['id']}",
                    use_container_width=True,
                ):
                    if saved:
                        st.session_state.saved.remove(item["id"])
                    else:
                        st.session_state.saved.add(item["id"])
                    st.rerun()

        st.write("")


def detail_box():
    item_id = st.session_state.selected_id

    if not item_id or item_id not in HERITAGE_BY_ID:
        return

    item = HERITAGE_BY_ID[item_id]

    st.markdown(
        f"""
        <div class="detail-box">
            <div class="detail-title">{tr(item,"title")}</div>
            <div class="detail-meta">{meta_tr(item, "museum")} · {meta_tr(item, "period")}</div>
            <div class="detail-text">{tr(item,"description")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def saved_section():
    st.markdown(
        f'<div class="section-title">{ui("my_saved_heritage")}</div>',
        unsafe_allow_html=True,
    )

    items = [
        HERITAGE_BY_ID[item_id]
        for item_id in st.session_state.saved
        if item_id in HERITAGE_BY_ID
    ]

    if not items:
        st.caption(ui("no_saved"))
        return

    cols = st.columns(min(4, len(items)))

    for i, item in enumerate(items):
        with cols[i % len(cols)]:
            st.markdown(
                f"""
                <div class="saved-card">
                    <div class="saved-title">{item.get("emoji","🏛️")} {tr(item,"title")}</div>
                    <div class="saved-meta">{meta_tr(item, "museum")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# HOME
# =========================================================
def home_page(ticket):
    st.markdown(
        f'<div class="section-title">{ui("heritage_guide")}</div>',
        unsafe_allow_html=True,
    )

    heritage_cards(ticket)
    detail_box()
    st.write("")
    saved_section()


# =========================================================
# MY HERITAGE
# =========================================================
def my_heritage_page(ticket):
    st.markdown(
        f'<div class="section-title">{ui("my_heritage")}</div>',
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)
    a.metric(ui("accessible"), len([x for x in ticket.get("access", []) if x in HERITAGE_BY_ID]))
    b.metric(ui("saved_count"), len(st.session_state.saved))
    c.metric(ui("viewed"), len(st.session_state.viewed))

    st.write("")
    st.markdown(f"#### {ui('recent_views')}")

    if not st.session_state.viewed:
        st.info(ui("no_recent_views"))
    else:
        for item_id in reversed(st.session_state.viewed[-6:]):
            item = HERITAGE_BY_ID[item_id]
            with st.container(border=True):
                st.write(f"{item.get('emoji','🏛️')} **{tr(item,'title')}**")
                st.caption(meta_tr(item, "museum"))

    st.write("")
    saved_section()


# =========================================================
# CITY HERITAGE
# =========================================================
def city_page(ticket):
    st.markdown(
        f'<div class="section-title">{ui("city_heritage")}</div>',
        unsafe_allow_html=True,
    )

    items = [
        HERITAGE_BY_ID[item_id]
        for item_id in ticket.get("access", [])
        if item_id in HERITAGE_BY_ID
        and HERITAGE_BY_ID[item_id].get("type") == "city"
    ]

    if not items:
        st.info(ui("no_city"))
        return

    df = pd.DataFrame([
        {"lat": item["lat"], "lon": item["lon"], "name": tr(item, "title")}
        for item in items
    ])

    st.map(df, latitude="lat", longitude="lon", size=150)

    for item in items:
        with st.expander(f"{item.get('emoji','🏛️')} {tr(item,'title')}"):
            st.write(tr(item, "description"))
            st.caption(meta_tr(item, "museum"))


# =========================================================
# SEARCH
# =========================================================
def search_page(ticket):
    st.markdown(
        f'<div class="section-title">{ui("search_heritage")}</div>',
        unsafe_allow_html=True,
    )

    query = st.text_input(
        ui("search_label"),
        placeholder=ui("search_placeholder"),
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
            searchable = " ".join([
                item.get("title", {}).get("ko", ""),
                item.get("title", {}).get("en", ""),
                item.get("title", {}).get("el", ""),
                item.get("subtitle", {}).get("ko", ""),
                item.get("subtitle", {}).get("en", ""),
                item.get("museum", ""),
                item.get("period", ""),
            ]).lower()

            if q in searchable:
                result.append(item)
    else:
        result = items

    st.caption(f"{len(result)}{ui('results')}")

    for item in result:
        with st.container(border=True):
            st.markdown(f"### {item.get('emoji','🏛️')} {tr(item,'title')}")
            st.write(tr(item, "subtitle"))
            st.caption(f"{meta_tr(item, 'museum')} · {meta_tr(item, 'period')}")

            if st.button(
                ui("open"),
                key=f"search_{item['id']}",
                type="primary",
            ):
                st.session_state.selected_id = item["id"]
                if item["id"] not in st.session_state.viewed:
                    st.session_state.viewed.append(item["id"])

    detail_box()


# =========================================================
# APP
# =========================================================
def main():
    if not st.session_state.authenticated:
        login_page()
        return

    ticket = TICKETS[st.session_state.ticket_no]

    header(ticket)

    tabs = st.tabs([
        ui("home"),
        ui("my_heritage_tab"),
        ui("city_heritage_tab"),
        ui("search_tab"),
    ])

    with tabs[0]:
        home_page(ticket)

    with tabs[1]:
        my_heritage_page(ticket)

    with tabs[2]:
        city_page(ticket)

    with tabs[3]:
        search_page(ticket)


main()
