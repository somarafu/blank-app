import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans # 클러스터링을 위해
import datetime

# --- 1. 앱 설정 ---
st.set_page_config(
    page_title="한국 시위 데이터 대시보드",
    page_icon="🇰🇷",
    layout="wide"
)

# --- 2. (중요) 데이터 로딩 ---
# @st.cache_data를 사용하면 데이터 로딩을 캐시하여 앱 속도를 높입니다.
@st.cache_data
def load_data(nrows):
    """
    이 함수는 실제 데이터를 로드하는 부분입니다.
    지금은 10,000개의 가상 시위 데이터를 생성합니다.
    
    *** 사용자가 실제 데이터를 가지고 있다면 이 함수 내부를 수정해야 합니다. ***
    
    예:
    try:
        data = pd.read_csv('your_protest_data.csv')
        # 'date' 컬럼이 문자열이라면 datetime으로 변환해야 합니다.
        data['date'] = pd.to_datetime(data['date_column_name'])
        # 'lat', 'lon' 컬럼이 있는지 확인하세요.
        
    except FileNotFoundError:
        st.error("데이터 파일을 찾을 수 없습니다. 'your_protest_data.csv'를 업로드하세요.")
        return pd.DataFrame()
        
    return data
    """
    
    # --- 가상 데이터 생성 시작 (실제 데이터로 이 부분을 교체하세요) ---
    N_ROWS = nrows
    DATE_COLUMN = 'date'
    
    # 한국의 주요 도시 중심 좌표 (서울, 부산, 광주)
    cities = {
        'Seoul': (37.5665, 126.9780),
        'Busan': (35.1796, 129.0756),
        'Gwangju': (35.1595, 126.8526)
    }
    
    data = []
    np.random.seed(42)
    
    for _ in range(N_ROWS):
        city_name = np.random.choice(list(cities.keys()), p=[0.6, 0.2, 0.2]) # 서울 60%, 부산/광주 20%
        lat, lon = cities[city_name]
        
        # 중심 좌표 근처에 무작위로 점 생성
        lat_offset = np.random.normal(0, 0.03) # 약 3.3km 반경
        lon_offset = np.random.normal(0, 0.03) # 약 3.3km 반경
        
        data.append({
            DATE_COLUMN: datetime.datetime(
                2024, 
                np.random.randint(1, 13), 
                np.random.randint(1, 28), 
                np.random.randint(0, 24), 
                np.random.randint(0, 60)
            ),
            'lat': lat + lat_offset,
            'lon': lon + lon_offset,
            'protest_type': np.random.choice(['노동', '시민', '환경', '개인']),
            'scale': np.random.choice(['소규모', '중규모', '대규모'], p=[0.5, 0.3, 0.2])
        })
    
    data = pd.DataFrame(data)
    data['hour'] = data[DATE_COLUMN].dt.hour
    # --- 가상 데이터 생성 종료 ---
    
    return data

# 클러스터링 시각화를 위한 색상 (10개)
CLUSTER_COLORS = ['#FF0000', '#0000FF', '#00FF00', '#FFFF00', '#00FFFF', 
                  '#FF00FF', '#FFA500', '#800080', '#008000', '#800000']


# --- 3. 사이드바 (필터) ---
st.sidebar.header("데이터 필터")

# 3-1. 시간 선택 슬라이더
hour_to_filter = st.sidebar.slider(
    '시간 선택:',
    min_value=0,
    max_value=23,
    value=17, # 기본값 17시
    step=1
)

# 3-2. 시위 유형 필터 (Multiselect)
all_types = ['노동', '시민', '환경', '개인']
types_to_filter = st.sidebar.multiselect(
    '시위 유형 선택:',
    options=all_types,
    default=all_types # 기본으로 모두 선택
)

# 3-3. 시위 규모 필터 (Multiselect)
all_scales = ['소규모', '중규모', '대규모']
scales_to_filter = st.sidebar.multiselect(
    '시위 규모 선택:',
    options=all_scales,
    default=all_scales # 기본으로 모두 선택
)

# 3-4. 클러스터 개수(K) 슬라이더 (뉴욕 예제와 동일)
k_clusters = st.sidebar.slider(
    '클러스터 개수 (K):',
    min_value=1,
    max_value=10,
    value=1, # 기본값 1 (클러스터링 없음)
    help='K=1은 클러스터링을 사용하지 않습니다. 2 이상을 선택하면 K-Means 클러스터링을 실행합니다.'
)


# --- 4. 데이터 로딩 및 필터링 ---
# 데이터 로드
data = load_data(10000)

# 필터 적용
filtered_data = data[
    (data['hour'] == hour_to_filter) &
    (data['protest_type'].isin(types_to_filter)) &
    (data['scale'].isin(scales_to_filter))
]

# --- 5. 메인 패널 (시각화) ---
st.title("🇰🇷 한국 시위 데이터 실시간 분석 대시보드")
st.markdown("이 대시보드는 가상의 시위 데이터를 사용하여 특정 시간대와 조건에 맞는 시위 발생 위치를 지도에 표시합니다.")

# 5-1. 맵 시각화 (클러스터링 포함)
subheader_text = f"시간: {hour_to_filter}:00, 선택된 시위 건수: {len(filtered_data)}건"
if k_clusters > 1:
    subheader_text += f" (K={k_clusters} 클러스터링 적용)"
st.subheader(subheader_text)


if not filtered_data.empty:
    if k_clusters > 1:
        # K=2 이상이면 K-Means 클러스터링 실행
        with st.spinner('위치 클러스터링 중...'):
            kmeans = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
            filtered_data['cluster'] = kmeans.fit_predict(filtered_data[['lat', 'lon']])
            
            # 클러스터 번호에 따라 색상 매핑
            filtered_data['color'] = filtered_data['cluster'].apply(
                lambda x: CLUSTER_COLORS[x % len(CLUSTER_COLORS)]
            )
            
            # 'color' 컬럼을 사용하여 지도에 색상 표시
            st.map(filtered_data, color='color')
            
    else:
        # K=1이면 (기본값) 클러스터링 없이 표시
        st.map(filtered_data)
        
else:
    st.warning("이 조건에 맞는 데이터가 없습니다.")

# 5-2. 시간대별 통계 (막대 차트)
st.subheader("전체 시간대별 시위 발생 건수")
# 원본 'data'를 사용해 전체 시간대별 히스토그램 생성
hist_values = np.histogram(data['hour'], bins=24, range=(0, 24))[0]
hist_df = pd.DataFrame({'Hour': range(24), 'Count': hist_values})
st.bar_chart(hist_df.set_index('Hour'))

# 5-3. 필터링된 원본 데이터 보기
if st.checkbox('필터링된 원본 데이터 보기'):
    st.subheader(f"{hour_to_filter}:00의 필터링된 데이터")
    st.dataframe(filtered_data.drop(['cluster', 'color'], errors='ignore'))
