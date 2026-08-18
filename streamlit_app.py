import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
import json, html

st.set_page_config(
    page_title="CULTURE PASS",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    heritage = json.loads((BASE_DIR / "data" / "heritage.json").read_text(encoding="utf-8"))
    tickets = json.loads((BASE_DIR / "data" / "tickets.json").read_text(encoding="utf-8"))
    return heritage, tickets

HERITAGE, TICKETS = load_data()
HERITAGE_BY_ID = {item["id"]: item for item in HERITAGE}

LANGS = {
    "한국어": ("ko", "ko-KR"),
    "English": ("en", "en-US"),
    "Ελληνικά": ("el", "el-GR"),
}

st.markdown("""
<style>
:root{
    --cp-bg:#FAF7F0;
    --cp-card:#FFFFFF;
    --cp-blue:#315FB5;
    --cp-blue2:#4A73C4;
    --cp-soft:#EFF6FF;
    --cp-brown:#764A38;
    --cp-line:#DCE5F3;
    --cp-text:#2E2A27;
}
html, body, [class*="css"] { font-family: "Pretendard","Noto Sans KR","Apple SD Gothic Neo",sans-serif; }
.stApp { background: var(--cp-bg); }
.block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 3rem; }
#MainMenu, footer, header { visibility: hidden; }

.cp-browser{
    background:white;
    border:1px solid #e4e7ec;
    border-radius:22px;
    box-shadow:0 15px 45px rgba(54,61,83,.08);
    overflow:hidden;
    margin-bottom:18px;
}
.cp-browserbar{
    height:44px; display:flex; align-items:center; gap:8px;
    padding:0 18px; border-bottom:1px solid #eceff3; background:#fff;
}
.dot{width:12px;height:12px;border-radius:50%;display:inline-block;}
.red{background:#e66a58}.yellow{background:#d7b851}.green{background:#62b65a}

.cp-brand-row{
    display:flex; justify-content:space-between; align-items:center;
    padding:16px 24px 10px 24px;
}
.cp-brand{
    font-family: Georgia, "Times New Roman", serif;
    font-weight:800; letter-spacing:.4px; color:#1f3765; font-size:25px;
}
.cp-badge{
    background:#eef4ff; color:var(--cp-blue); border-radius:999px;
    padding:8px 14px; font-weight:700; font-size:13px;
}
.cp-verified{
    margin:0 24px 18px 24px; padding:18px 20px;
    background:linear-gradient(90deg,#f0f5ff,#fbfcff);
    border:1px solid #d9e5fa; border-radius:14px;
    display:flex; align-items:center; gap:14px;
}
.cp-verified .icon{
    width:48px;height:48px;border-radius:50%;background:var(--cp-blue);
    display:flex;align-items:center;justify-content:center;color:white;font-size:24px;
}
.cp-verified strong{color:var(--cp-blue);font-size:18px;}
.cp-muted{color:#777;font-size:13px;}

.cp-feature{
    background:#fff;border:1px solid #e7e9ee;border-radius:20px;
    padding:22px 20px; min-height:205px;
    box-shadow:0 8px 24px rgba(45,55,80,.045);
}
.cp-feature h4{margin:0;color:var(--cp-blue);font-size:19px;}
.cp-feature p{margin:.25rem 0 0 0;color:#555;line-height:1.55;font-size:14px;}
.cp-feature .ic{
    width:48px;height:48px;border-radius:50%;background:#345eb1;color:white;
    display:flex;align-items:center;justify-content:center;font-size:23px;flex:0 0 auto;
}
.cp-feature-row{display:flex;gap:14px;align-items:flex-start;margin-bottom:18px;}
.cp-card{
    background:#fff;border:1px solid #e7eaf0;border-radius:16px;
    padding:16px; min-height:315px;
}
.cp-card-title{color:var(--cp-blue);font-size:20px;font-weight:800;margin-bottom:3px;}
.cp-card-sub{color:#474747;font-size:13px;min-height:42px;line-height:1.45;}
.cp-art{
    height:138px;border-radius:12px;background:linear-gradient(135deg,#e8eef9,#f5eee8);
    display:flex;align-items:center;justify-content:center;font-size:74px;
    margin:10px 0 12px 0;border:1px solid #e6e8ee;
}
.cp-detail{
    background:#fff;border:1px solid #dce4f2;border-radius:18px;padding:20px;
    margin-top:15px;
}
.cp-kicker{color:var(--cp-brown);font-weight:700;font-size:14px;letter-spacing:.02em;}
.cp-h1{
    font-family: Georgia,"Times New Roman","Noto Serif KR",serif;
    font-size:38px;font-weight:800;color:#201d1b;margin:.1rem 0 .3rem 0;
}
.cp-h2{font-size:23px;font-weight:800;color:#253c69;margin-bottom:6px;}
.cp-quote{
    background:#eef4ff;border:1px solid #dae7fb;border-radius:16px;
    padding:18px 22px;color:#315fb5;font-weight:750;text-align:center;font-size:16px;
}
.cp-section-title{
    font-size:21px;font-weight:800;color:#315fb5;margin:8px 0 12px 0;
}
.cp-saved-chip{
    border:1px solid #dfe5ef;border-radius:12px;padding:10px 12px;background:white;
    min-height:68px;
}
.cp-login{
    max-width:650px;margin:7vh auto 0 auto;background:white;border:1px solid #e7e5e1;
    border-radius:26px;padding:34px 38px;box-shadow:0 18px 60px rgba(70,60,50,.10);
}
.cp-login .logo{
    font-family:Georgia,serif;color:#274b91;font-weight:800;font-size:36px;
}
.cp-login .desc{color:#666;line-height:1.65;margin:8px 0 20px 0;}
div[data-testid="stButton"] button{
    border-radius:10px;
    border:1px solid #c9d7ef;
    font-weight:700;
}
div[data-testid="stButton"] button[kind="primary"]{
    background:#315fb5;border-color:#315fb5;color:white;
}
div[data-testid="stTextInput"] input{
    border-radius:10px;
}
div[data-testid="stTabs"] button{font-weight:750;}
</style>
""", unsafe_allow_html=True)

def init_state():
    defaults = {
        "authenticated": False,
        "ticket_no": None,
        "saved": set(),
        "viewed": [],
        "selected_id": None,
        "lang_label": "한국어",
        "audio_count": 0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def t(item, field):
    code = LANGS[st.session_state.lang_label][0]
    return item[field].get(code) or item[field].get("en") or next(iter(item[field].values()))

def speech_button(text, key):
    lang_code = LANGS[st.session_state.lang_label][1]
    safe_text = json.dumps(text)
    components.html(f"""
    <button onclick="speak_{key}()" style="
        width:100%;height:38px;border-radius:9px;border:1px solid #b8c8e6;
        background:white;color:#315fb5;font-weight:700;cursor:pointer;font-size:14px;">
        🔊 Audio
    </button>
    <script>
      function speak_{key}(){{
        try {{
          const u = new SpeechSynthesisUtterance({safe_text});
          u.lang = "{lang_code}";
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(u);
        }} catch(e) {{}}
      }}
    </script>
    """, height=44)

def login_screen():
    st.markdown('<div class="cp-login">', unsafe_allow_html=True)
    st.markdown('<div class="logo">CULTURE PASS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="desc">박물관 입장권의 <b>고객번호</b>로 인증한 뒤, '
        '관람한 문화유산의 다국어 해설·음성 안내·도시 문화유산 연계 정보를 다시 확인하는 프로토타입입니다.</div>',
        unsafe_allow_html=True
    )
    customer_no = st.text_input("티켓 고객번호", placeholder="예: CP-ATH-0820-001")
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("CULTURE PASS 입장", type="primary", use_container_width=True):
            key = customer_no.strip().upper()
            if key in TICKETS:
                st.session_state.authenticated = True
                st.session_state.ticket_no = key
                st.session_state.saved = set()
                st.session_state.viewed = []
                st.session_state.selected_id = None
                st.rerun()
            else:
                st.error("유효하지 않은 고객번호입니다.")
    with c2:
        if st.button("데모 고객번호 입력", use_container_width=True):
            st.session_state["demo_code_hint"] = "CP-ATH-0820-001"
    if st.session_state.get("demo_code_hint"):
        st.info("데모 고객번호: **CP-ATH-0820-001**")
    st.caption("※ 학교 발표용 프로토타입입니다. 실제 서비스에서는 고객번호를 서버 DB에서 암호화·검증해야 합니다.")
    st.markdown('</div>', unsafe_allow_html=True)

def top_header(ticket):
    st.markdown('<div class="cp-kicker">CULTURE PASS · DIGITAL HERITAGE GUIDE</div>', unsafe_allow_html=True)
    h1, controls = st.columns([3.4, 1.6], vertical_alignment="center")
    with h1:
        st.markdown('<div class="cp-h1">관람 이후에도 이어지는 문화유산 경험</div>', unsafe_allow_html=True)
    with controls:
        cc1, cc2 = st.columns([1.5,1])
        with cc1:
            lang = st.selectbox("Language", list(LANGS.keys()),
                                index=list(LANGS.keys()).index(st.session_state.lang_label),
                                label_visibility="collapsed")
            st.session_state.lang_label = lang
        with cc2:
            if st.button("로그아웃", use_container_width=True):
                for key in ["authenticated","ticket_no","saved","viewed","selected_id"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    st.markdown(
        f'<div class="cp-browser"><div class="cp-browserbar">'
        f'<span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span>'
        f'</div><div class="cp-brand-row"><div class="cp-brand">CULTURE PASS</div>'
        f'<div class="cp-badge">{html.escape(ticket["route"])}</div></div>'
        f'<div class="cp-verified"><div class="icon">🎟</div><div>'
        f'<strong>Ticket Verified ✓</strong><br>'
        f'<span>{html.escape(ticket["customer_name"])} · 방문 {ticket["visit_date"]} · 접근기한 {ticket["valid_until"]}</span>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

def feature_panel():
    left, right = st.columns(2)
    features = [
        ("🎟", "티켓 기반 인증", "입장권 고객번호를 기반으로 승인된 문화유산 콘텐츠에만 접근합니다."),
        ("🌐", "다국어 UI · 해설", "한국어·영어·그리스어로 텍스트와 음성 해설을 전환합니다."),
        ("🔊", "음성·텍스트 아카이브", "관람 당시의 해설을 관람 이후에도 재생·재탐색할 수 있습니다."),
        ("🏛", "박물관·도시 문화유산 연계", "박물관 밖 지하철역·공공공간의 문화유산까지 하나의 탐색 경험으로 연결합니다.")
    ]
    for col, pair in zip([left,right,left,right], features):
        icon, title, desc = pair
        with col:
            st.markdown(f"""
            <div class="cp-feature-row">
                <div class="ic">{icon}</div>
                <div><h4>{title}</h4><p>{desc}</p></div>
            </div>
            """, unsafe_allow_html=True)

def heritage_cards(access_ids):
    access_items = [HERITAGE_BY_ID[x] for x in access_ids if x in HERITAGE_BY_ID]
    cols = st.columns(min(3, len(access_items)))
    for i, item in enumerate(access_items):
        col = cols[i % len(cols)]
        with col:
            st.markdown(
                f'<div class="cp-card">'
                f'<div class="cp-card-title">{html.escape(t(item,"title"))}</div>'
                f'<div class="cp-card-sub">{html.escape(t(item,"subtitle"))}</div>'
                f'<div class="cp-art">{item["emoji"]}</div>'
                f'<div class="cp-muted">{html.escape(item["museum"])}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            b1, b2 = st.columns([1,1])
            with b1:
                speech_button(t(item,"description"), item["id"])
            with b2:
                if st.button("Read More", key=f"read_{item['id']}", use_container_width=True):
                    st.session_state.selected_id = item["id"]
                    if item["id"] not in st.session_state.viewed:
                        st.session_state.viewed.append(item["id"])
            saved = item["id"] in st.session_state.saved
            label = "★ 저장됨" if saved else "☆ My Heritage 저장"
            if st.button(label, key=f"save_{item['id']}", use_container_width=True):
                if saved:
                    st.session_state.saved.remove(item["id"])
                else:
                    st.session_state.saved.add(item["id"])
                st.rerun()

def detail_panel():
    item_id = st.session_state.selected_id
    if not item_id or item_id not in HERITAGE_BY_ID:
        return
    item = HERITAGE_BY_ID[item_id]
    st.markdown(
        f'<div class="cp-detail"><div class="cp-h2">{html.escape(t(item,"title"))}</div>'
        f'<div class="cp-muted">{html.escape(item["museum"])} · {html.escape(item["period"])}</div>'
        f'<p style="line-height:1.75;font-size:15px;">{html.escape(t(item,"description"))}</p>'
        f'</div>', unsafe_allow_html=True
    )

def saved_section():
    st.markdown('<div class="cp-section-title">My Saved Heritage</div>', unsafe_allow_html=True)
    saved = [HERITAGE_BY_ID[x] for x in st.session_state.saved if x in HERITAGE_BY_ID]
    if not saved:
        st.caption("저장한 문화유산이 없습니다. 카드의 ‘My Heritage 저장’을 눌러보세요.")
        return
    cols = st.columns(min(4, len(saved)))
    for idx, item in enumerate(saved):
        with cols[idx % len(cols)]:
            st.markdown(
                f'<div class="cp-saved-chip"><b>{item["emoji"]} {html.escape(t(item,"title"))}</b>'
                f'<br><span class="cp-muted">{html.escape(item["museum"])}</span></div>',
                unsafe_allow_html=True
            )

def dashboard_home(ticket):
    st.markdown('<div class="cp-section-title">오늘의 CULTURE PASS</div>', unsafe_allow_html=True)
    heritage_cards(ticket["access"])
    detail_panel()
    saved_section()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="cp-quote">“정보 설계의 핵심은 데이터를 나열하는 데 있지 않고, 사용자가 맥락 속에서 지식을 재구성하도록 지원하는 데 있다.”</div>', unsafe_allow_html=True)

def my_heritage(ticket):
    st.markdown('<div class="cp-section-title">나의 관람 기록</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("접근 가능한 문화유산", len(ticket["access"]))
    c2.metric("저장한 문화유산", len(st.session_state.saved))
    c3.metric("상세 열람 기록", len(st.session_state.viewed))
    st.markdown("#### 최근 열람")
    if not st.session_state.viewed:
        st.info("아직 상세 열람 기록이 없습니다.")
    else:
        for item_id in reversed(st.session_state.viewed[-5:]):
            item = HERITAGE_BY_ID[item_id]
            st.write(f"{item['emoji']} **{t(item,'title')}** — {item['museum']}")
    st.markdown("#### 저장한 문화유산")
    saved_section()

def city_heritage(ticket):
    items = [HERITAGE_BY_ID[x] for x in ticket["access"]
             if x in HERITAGE_BY_ID and HERITAGE_BY_ID[x]["type"] == "city"]
    st.markdown('<div class="cp-section-title">City Heritage · 도시 속 문화유산</div>', unsafe_allow_html=True)
    st.write("박물관 관람 경험을 지하철역·고고학 유적·공공공간의 문화유산 탐색으로 확장합니다.")
    if not items:
        st.info("현재 티켓에는 도시 문화유산 경로가 포함되어 있지 않습니다.")
        return
    df = pd.DataFrame([{
        "lat": x["lat"], "lon": x["lon"],
        "name": t(x,"title")
    } for x in items])
    st.map(df, latitude="lat", longitude="lon", size=180)
    for item in items:
        with st.expander(f"{item['emoji']} {t(item,'title')}"):
            st.write(t(item,"description"))
            st.caption(item["museum"])

def search_page(ticket):
    st.markdown('<div class="cp-section-title">문화유산 탐색</div>', unsafe_allow_html=True)
    query = st.text_input("검색", placeholder="예: 파르테논, IT Museum, Metro")
    items = [HERITAGE_BY_ID[x] for x in ticket["access"] if x in HERITAGE_BY_ID]
    if query.strip():
        q = query.strip().lower()
        filtered = []
        for item in items:
            blob = " ".join([
                item["title"].get("ko",""), item["title"].get("en",""), item["title"].get("el",""),
                item["subtitle"].get("ko",""), item["subtitle"].get("en",""), item["museum"], item["period"]
            ]).lower()
            if q in blob:
                filtered.append(item)
    else:
        filtered = items
    st.caption(f"{len(filtered)}개 결과")
    for item in filtered:
        with st.container(border=True):
            st.markdown(f"### {item['emoji']} {t(item,'title')}")
            st.write(t(item,"subtitle"))
            st.caption(f"{item['museum']} · {item['period']}")
            if st.button("상세 보기", key=f"search_{item['id']}"):
                st.session_state.selected_id = item["id"]
                if item["id"] not in st.session_state.viewed:
                    st.session_state.viewed.append(item["id"])
                st.rerun()
    detail_panel()

def app():
    if not st.session_state.authenticated:
        login_screen()
        return

    ticket = TICKETS[st.session_state.ticket_no]
    top_header(ticket)

    # 핵심 기능은 첫 화면에서도 한눈에 보이도록 유지
    with st.expander("CULTURE PASS 핵심 기능 보기", expanded=False):
        feature_panel()

    tabs = st.tabs(["HOME", "MY HERITAGE", "CITY HERITAGE", "SEARCH"])
    with tabs[0]:
        dashboard_home(ticket)
    with tabs[1]:
        my_heritage(ticket)
    with tabs[2]:
        city_heritage(ticket)
    with tabs[3]:
        search_page(ticket)

app()
