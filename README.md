CULTURE PASS — Streamlit Prototype

그리스 문화유산 탐방 경험을 바탕으로 만든 티켓 인증형 다국어 문화유산 안내 플랫폼 예시입니다.

실행

pip install -r requirements.txt
streamlit run streamlit_app.py

데모 고객번호

CP-ATH-0820-001 : 전체 기능

CP-MUS-0821-002 : Hellenic IT Museum만

CP-ATH-0822-003 : 아테네 문화유산 경로

구현된 기능

티켓 고객번호 기반 접근 제한

한국어 / English / Ελληνικά 전환

브라우저 Web Speech API 기반 음성 해설

Read More 상세 해설

My Heritage 저장

관람 이력

도시 문화유산 지도

통합 검색

티켓별 접근 권한 분리

실제 서비스로 확장할 때

SQLite/PostgreSQL 사용자·티켓 DB

고객번호 해시/서버 검증

박물관 발권 시스템 API 연동

실제 문화재 이미지와 라이선스 관리

전문 성우/TTS 오디오 파일 저장

관리자 CMS

접근 만료 정책, 개인정보 처리, 로그 관리

데이터 수정

data/tickets.json, data/heritage.json을 수정하면 코드 구조를 크게 바꾸지 않고 예시 데이터를 교체할 수 있습니다.

※ 현재 이미지는 발표용 프로토타입이므로 카드 중앙을 이모지 기반 시각 요소로 구성했습니다.
실제 촬영 사진을 넣을 때는 heritage.json에 image 경로 필드를 추가한 뒤 st.image()로 교체하면 됩니다.
