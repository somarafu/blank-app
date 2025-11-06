import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans # 클러스터링을 위해
import datetime
import plotly.express as px # Plotly (도넛 차트)
import pydeck as pdk # PyDeck (히트맵)

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
    # + 국가별 데이터 분산 범위를 위한 'scale' 추가
    cities = {
        'Paris': {'coords': (48.8566, 2.3522), 'country': 'France', 'bias': [0.5, 0.3, 0.1, 0.1], 'scale': 2.5},
        'Berlin': {'coords': (52.5200, 13.4050), 'country': 'Germany', 'bias': [0.2, 0.3, 0.4, 0.1], 'scale': 2.0},
        'London': {'coords': (51.5074, -0.1278), 'country': 'UK', 'bias': [0.3, 0.4, 0.2, 0.1], 'scale': 2.5},
        'Madrid': {'coords': (40.4168, -3.7038), 'country': 'Spain', 'bias': [0.4, 0.3, 0.2, 0.1], 'scale': 3.0},
        'Rome': {'coords': (41.9028, 12.4964), 'country': 'Italy', 'bias': [0.4, 0.4, 0.1, 0.1], 'scale': 3.0},
        'Moscow': {'coords': (55.7558, 37.6173), 'country': 'Russia', 'bias': [0.2, 0.5, 0.1, 0.2], 'scale': 5.0}, # 넓은 범위
        'Istanbul': {'coords': (41.0082, 28.9784), 'country': 'Turkey', 'bias': [0.3, 0.5, 0.1, 0.1], 'scale': 3.5},
        'Delhi': {'coords': (28.6139, 77.2090), 'country': 'India', 'bias': [0.4, 0.4, 0.1, 0.1], 'scale': 4.0},
        'Beijing': {'coords': (39.9042, 116.4074), 'country': 'China', 'bias': [0.5, 0.2, 0.2, 0.1], 'scale': 4.5},
        'Tokyo': {'coords': (35.6895, 139.6917), 'country': 'Japan', 'bias': [0.1, 0.6, 0.2, 0.1], 'scale': 1.5}, # 좁은 범위
        'Seoul': {'coords': (37.5665, 126.9780), 'country': 'South Korea', 'bias': [0.4, 0.3, 0.1, 0.2], 'scale': 1.0} # 가장 좁은 범위
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
        country_scale = city_info['scale'] # 국가별 스케일 값 가져오기
        
        # 중심 좌표 근처에 무작위로 점 생성 (국가별 스케일 적용)
        # np.random.normal (정규분포) 대신 np.random.uniform (균등분포) 사용
        lat_offset = np.random.uniform(-country_scale, country_scale) 
        lon_offset = np.random.uniform(-country_scale, country_scale)
        
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

# --- 3-5. 시각화 옵션 ---
st.sidebar.subheader("지도 시각화 옵션")
map_viz_type = st.sidebar.selectbox(
    "지도 유형 선택:",
    options=['점 지도 (Clustering)', '밀도 지도 (Heatmap)'],
    index=0
)

# 3-6. 클러스터 개수(K) 슬라이더
k_clusters = 1
if map_viz_type == '점 지도 (Clustering)':
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

# --- 5-1. 요약 통계 (Metrics) ---
st.subheader("필터 요약 (At a Glance)")

col_m1, col_m2, col_m3 = st.columns(3)

# 1. 총 시위 건수
col_m1.metric(label="총 시위 건수", value=f"{len(filtered_data)} 건")

# 2. 최다 발생 국가
if not filtered_data.empty:
    top_country = filtered_data['country'].value_counts().idxmax()
    top_country_count = filtered_data['country'].value_counts().max()
    col_m2.metric(label="최다 발생 국가", value=top_country, help=f"{top_country}에서 {top_country_count}건 발생")
else:
    col_m2.metric(label="최다 발생 국가", value="데이터 없음")

# 3. 최다 시위 유형
if not filtered_data.empty:
    top_type = filtered_data['protest_type'].value_counts().idxmax()
    top_type_count = filtered_data['protest_type'].value_counts().max()
    col_m3.metric(label="최다 시위 유형", value=top_type, help=f"{top_type} 유형 {top_type_count}건 발생")
else:
    col_m3.metric(label="최다 시위 유형", value="데이터 없음")
    
st.divider() # 구분선 추가


# 5-1. 맵 시각화 (클러스터링 또는 히트맵)
# 기존 subheader_text의 총 건수 정보는 위 metric으로 이동했습니다.
map_subheader = f"시위 발생 위치 지도 ({map_viz_type})"
if map_viz_type == '점 지도 (Clustering)' and k_clusters > 1:
    map_subheader += f" (K={k_clusters} 클러스터링 적용)"
st.subheader(map_subheader)


if not filtered_data.empty:
    if map_viz_type == '점 지도 (Clustering)':
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
            
    elif map_viz_type == '밀도 지도 (Heatmap)':
        # PyDeck을 사용한 히트맵
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=filtered_data['lat'].mean(),
                longitude=filtered_data['lon'].mean(),
                zoom=3,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                   'HeatmapLayer',
                   data=filtered_data[['lat', 'lon']],
                   get_position='[lon, lat]',
                   opacity=0.9,
                   radius_pixels=70,
                   intensity=1,
                   threshold=0.03,
                ),
            ],
        ))
        
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
    # 5-2-2. 유형별 시위 건수 (도넛 차트)
    st.subheader("시위 유형별 건수 (비율)")
    if not filtered_data.empty:
        type_counts = filtered_data['protest_type'].value_counts().reset_index()
        type_counts.columns = ['type', 'count'] # 컬럼명 변경
        
        fig_pie = px.pie(
            type_counts, 
            values='count', 
            names='type', 
            hole=0.4, # 도넛 차트
            color_discrete_sequence=px.colors.sequential.Purples_r # 색상 테마
        )
        fig_pie.update_layout(
            legend_title_text='시위 유형',
            legend_orientation='h', # 범례 가로로 표시
            legend_y=-0.2
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("데이터 없음")

# 5-3. 국가별 시위 유형 분석 (누적 막대 차트)
st.subheader("국가별 시위 유형 분석")
if not filtered_data.empty:
    # 국가(index) vs 유형(columns)으로 피벗 테이블 생성
    pivot_df = filtered_data.pivot_table(
        index='country', 
        columns='protest_type', 
        aggfunc='size', 
        fill_value=0
    )
    # 누적 막대 차트
    st.bar_chart(pivot_df)
else:
    st.info("데이터 없음")

# 5-4. 필터링된 원본 데이터 보기
if st.checkbox('필터링된 원본 데이터 보기'):
    st.subheader("필터링된 데이터 (최대 1,000건 표시)")
    # 'cluster', 'color' 컬럼이 없을 수도 있으므로 errors='ignore' 사용
    st.dataframe(filtered_data.drop(['cluster', 'color'], axis=1, errors='ignore').head(1000))
