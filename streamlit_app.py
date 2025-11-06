import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans # 클러스터링을 위해
import datetime

# --- 1. 앱 설정 ---
st.set_page_config(
    page_title="유라시아 시위 데이터 대시보드",
    page_icon="🌏",
    layout="wide"
)

# --- 2. (중요) 데이터 로딩 ---
# @st.cache_data를 사용하면 데이터 로딩을 캐시하여 앱 속도를 높입니다.
@st.cache_data
def load_data(nrows):
    """
    이 함수는 유라시아 대륙의 가상 시위 데이터를 생성합니다.
    (2024.01.01 ~ 2025.11.06)
    
    *** 이 코드는 실제 데이터가 아닌, 시뮬레이션된 데이터입니다. ***
    *** 국가별로 시위 유형에 대한 가상의 편향(bias)이 적용되었습니다. ***
    
    데이터 구조: [date, lat, lon, country, protest_type, scale]
    """
    
    N_ROWS = nrows
    DATE_COLUMN = 'date'
    
    # 유라시아 주요 도시 및 국가, 좌표, (가상) 시위 유형 편향
    # 편향(bias) 순서: ['노동', '시민', '환경', '개인']
    cities = {
        'Paris': {'coords': (48.8566, 2.3522), 'country': 'France', 'bias': [0.5, 0.3, 0.1, 0.1]},
        'Berlin': {'coords': (52.5200, 13.4050), 'country': 'Germany', 'bias': [0.2, 0.3, 0.4, 0.1]},
        'London': {'coords': (51.5074, -0.1278), 'country': 'UK', 'bias': [0.3, 0.4, 0.2, 0.1]},
        'Madrid': {'coords': (40.4168, -3.7038), 'country': 'Spain', 'bias': [0.4, 0.3, 0.2, 0.1]},
        'Rome': {'coords': (41.9028, 12.4964), 'country': 'Italy', 'bias': [0.4, 0.4, 0.1, 0.1]},
        'Moscow': {'coords': (55.7558, 37.6173), 'country': 'Russia', 'bias': [0.2, 0.5, 0.1, 0.2]},
        'Istanbul': {'coords': (41.0082, 28.9784), 'country': 'Turkey', 'bias': [0.3, 0.5, 0.1, 0.1]},
        'Delhi': {'coords': (28.6139, 77.2090), 'country': 'India', 'bias': [0.4, 0.4, 0.1, 0.1]},
        'Beijing': {'coords': (39.9042, 116.4074), 'country': 'China', 'bias': [0.5, 0.2, 0.2, 0.1]},
        'Tokyo': {'coords': (35.6895, 139.6917), 'country': 'Japan', 'bias': [0.1, 0.6, 0.2, 0.1]},
        'Seoul': {'coords': (37.5665, 126.9780), 'country': 'South Korea', 'bias': [0.4, 0.3, 0.1, 0.2]},
    }
    
    city_names = list(cities.keys())
    data = []
    np.random.seed(42)
    
    # 날짜 범위 설정 (2024-01-01 부터 2025-11-06 까지)
    start_timestamp = datetime.datetime(2024, 1, 1).timestamp()
    end_timestamp = datetime.datetime(2025, 11, 6).timestamp() # 현재 날짜

    for _ in range(N_ROWS):
        # 무작위 도시 선택 (일부 도시가 더 자주 선택되도록 가중치 부여 가능)
        city_name = np.random.choice(city_names)
        city_info = cities[city_name]
        
        lat, lon = city_info['coords']
        country = city_info['country']
        bias = city_info['bias']
        
        # 중심 좌표 근처에 무작위로 점 생성
        lat_offset = np.random.normal(0, 0.05) 
        lon_offset = np.random.normal(0, 0.05)
        
        # 무작위 날짜 생성
        random_timestamp = np.random.uniform(start_timestamp, end_timestamp)
        random_date = datetime.datetime.fromtimestamp(random_timestamp)
        
        data.append({
            DATE_COLUMN: random_date,
            'lat': lat + lat_offset,
            'lon': lon + lon_offset,
            'country': country,
            'protest_type': np.random.choice(['노동', '시민', '환경', '개인'], p=bias),
            'scale': np.random.choice(['소규모', '중규모', '대규모'], p=[0.5, 0.3, 0.2])
        })
    
    data = pd.DataFrame(data)
    
    # 'date' 컬럼을 datetime 객체로 변환 (중복 확인)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    
    return data

# 클러스터링 시각화를 위한 색상 (10개)
CLUSTER_COLORS = ['#FF0000', '#0000FF', '#00FF00', '#FFFF00', '#00FFFF', 
                  '#FF00FF', '#FFA500', '#800080', '#008000', '#800000']


# --- 3. 사이드바 (필터) ---
st.sidebar.header("데이터 필터 (Filters)")

# 3-1. 날짜 범위 선택 (Date Range)
st.sidebar.subheader("날짜 필터")
min_date = datetime.date(2024, 1, 1)
max_date = datetime.date(2025, 11, 6) # 오늘 날짜

start_date = st.sidebar.date_input(
    '시작일 (Start Date)', 
    min_date,
    min_value=min_date,
    max_value=max_date
)
end_date = st.sidebar.date_input(
    '종료일 (End Date)', 
    max_date,
    min_value=start_date, # 시작일보다 빠를 수 없음
    max_value=max_date
)

# --- 데이터 로딩 ---
# (필터 옵션을 채우기 위해 필터보다 먼저 로드)
data = load_data(20000) # 데이터 양을 20,000건으로 늘림

# 3-2. 국가 선택 (Country)
st.sidebar.subheader("국가 필터")
all_countries = sorted(data['country'].unique())
countries_to_filter = st.sidebar.multiselect(
    '국가 선택:',
    options=all_countries,
    default=all_countries # 기본으로 모두 선택
)

# 3-3. 시위 유형 필터 (Protest Type)
st.sidebar.subheader("시위 유형 필터")
all_types = ['노동', '시민', '환경', '개인']
types_to_filter = st.sidebar.multiselect(
    '시위 유형 선택:',
    options=all_types,
    default=all_types # 기본으로 모두 선택
)

# 3-4. 시위 규모 필터 (Protest Scale)
st.sidebar.subheader("시위 규모 필터")
all_scales = ['소규모', '중규모', '대규모']
scales_to_filter = st.sidebar.multiselect(
    '시위 규모 선택:',
    options=all_scales,
    default=all_scales # 기본으로 모두 선택
)

# 3-5. 클러스터 개수(K) 슬라이더
st.sidebar.subheader("클러스터링")
k_clusters = st.sidebar.slider(
    '클러스터 개수 (K):',
    min_value=1,
    max_value=10,
    value=1, # 기본값 1 (클러스터링 없음)
    help='K=1은 클러스터링을 사용하지 않습니다. 2 이상을 선택하면 K-Means 클러스터링을 실행합니다.'
)


# --- 4. 데이터 필터링 ---
# 날짜 필터링을 위해 datetime.date를 datetime.datetime으로 변환
start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
end_datetime = datetime.datetime.combine(end_date, datetime.time.max)

filtered_data = data[
    (data['date'] >= start_datetime) &
    (data['date'] <= end_datetime) &
    (data['country'].isin(countries_to_filter)) &
    (data['protest_type'].isin(types_to_filter)) &
    (data['scale'].isin(scales_to_filter))
]

# --- 5. 메인 패널 (시각화) ---
st.title("🌏 유라시아 대륙 시위 데이터 분석 대시보드 (2024-2025)")
st.markdown(f"**분석 기간:** `{start_date.isoformat()}` 부터 `{end_date.isoformat()}` 까지. (이 대시보드는 가상의 시뮬레이션 데이터입니다.)")

# 5-1. 맵 시각화 (클러스터링 포함)
subheader_text = f"필터링된 총 시위 건수: **{len(filtered_data)}**건"
if k_clusters > 1:
    subheader_text += f" (K={k_clusters} 클러스터링 적용)"
st.subheader(subheader_text)


if not filtered_data.empty:
    if k_clusters > 1:
        # K=2 이상이면 K-Means 클러스터링 실행
        with st.spinner('위치 클러스터링 중...'):
            kmeans = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
            # copy()를 사용하여 SettingWithCopyWarning 방지
            filtered_data_copy = filtered_data.copy()
            filtered_data_copy['cluster'] = kmeans.fit_predict(filtered_data_copy[['lat', 'lon']])
            
            # 클러스터 번호에 따라 색상 매핑
            filtered_data_copy['color'] = filtered_data_copy['cluster'].apply(
                lambda x: CLUSTER_COLORS[x % len(CLUSTER_COLORS)]
            )
            
            # 'color' 컬럼을 사용하여 지도에 색상 표시
            st.map(filtered_data_copy, color='color')
            
    else:
        # K=1이면 (기본값) 클러스터링 없이 표시
        st.map(filtered_data)
        
else:
    st.warning("이 조건에 맞는 데이터가 없습니다. 필터를 조정해 주세요.")

# 5-2. 통계 차트 (2단 컬럼)
col1, col2 = st.columns(2)

with col1:
    # 5-2-1. 국가별 시위 건수
    st.subheader("국가별 시위 건수")
    if not filtered_data.empty:
        country_counts = filtered_data['country'].value_counts()
        st.bar_chart(country_counts)
    else:
        st.info("데이터 없음")

with col2:
    # 5-2-2. 유형별 시위 건수
    st.subheader("시위 유형별 건수")
    if not filtered_data.empty:
        type_counts = filtered_data['protest_type'].value_counts()
        st.bar_chart(type_counts)
    else:
        st.info("데이터 없음")

# 5-3. 기간별 시위 발생 추이 (Line Chart)
st.subheader("기간별 시위 발생 추이")
if not filtered_data.empty:
    # 'date' 컬럼을 인덱스로 설정하고, 일별(D)로 리샘플링하여 개수 집계
    timeline_data = filtered_data.set_index('date').resample('D').size().reset_index(name='Count')
    st.line_chart(timeline_data.set_index('date'))
else:
    st.info("데이터 없음")

# 5-4. 필터링된 원본 데이터 보기
if st.checkbox('필터링된 원본 데이터 보기'):
    st.subheader("필터링된 데이터 (최대 1,000건 표시)")
    # 'cluster', 'color' 컬럼이 없을 수도 있으므로 errors='ignore' 사용
    st.dataframe(filtered_data.drop(['cluster', 'color'], axis=1, errors='ignore').head(1000))
