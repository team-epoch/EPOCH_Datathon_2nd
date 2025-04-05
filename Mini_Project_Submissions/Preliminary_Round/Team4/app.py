import streamlit as st
import pandas as pd
import os
import json
import openai
import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ml_recommendation 모듈에서 필요한 함수들 import
from ml_recommendation import (
    parse_user_request,
    select_books_for_recommendation,
    get_candidates_for_user,
    build_recommendation_system,
    create_tfidf_features,
    train_genre_prediction_model
)

try:
    from config import OPENAI_API_KEY
    openai.api_key = OPENAI_API_KEY
except ImportError:
    openai.api_key = os.environ.get("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=openai.api_key)

# 추천 시스템 함수
@st.cache_resource
def load_recommendation_system(processed_data_path):
    """
    머신러닝 기반 추천 시스템 로드 (캐싱 적용)
    """
    return build_recommendation_system(processed_data_path)

def create_recommendation_prompt(user_request, top_candidates_df):
    """
    사용자 요청과 후보 도서 정보를 바탕으로 LLM 프롬프트 생성
    """
    candidates_text = ""
    
    for i, (_, book) in enumerate(top_candidates_df.iterrows(), 1):
        candidates_text += f"후보 {i}:\n"
        candidates_text += f"제목: {book['title']}\n"
        candidates_text += f"저자: {book['main_author']}\n"
        candidates_text += f"출판사: {book['publisher']}\n"
        candidates_text += f"카테고리: {book['estimated_category']}\n"
        candidates_text += f"대상 독자: {book['age_group']}\n"
        
        # 설명이 너무 길면 짧게 잘라서 표시
        description = book['description']
        if isinstance(description, str) and len(description) > 200:
            description = description[:200] + "..."
        candidates_text += f"설명: {description}\n\n"
    
    prompt = f"""
    사용자 요청: {user_request}

    아래는 사용자 요청에 기반하여 선별된 도서 후보 목록입니다:

    {candidates_text}

    이제 다음 조건을 충족하도록 답변을 생성해주세요:

    1. 위 도서 목록 중에서 **사용자 요청(반드시 제목을 보고 과학과 관련한 책을 골라주세요)에 가장 적합한 3권**을 선택하세요.
    2. 각 도서에 대해 다음 항목을 포함해 상세하게 작성하세요:
    - 추천 이유: **사용자의 니즈와 요청 배경에 적합한 이유를 구체적으로 설명**해주세요.
    - 주요 내용: 도서가 다루는 **핵심 개념, 주요 챕터, 대표 사례, 저자의 관점** 등을 간결하고 명확하게 정리해주세요.
    - 이런 분께 추천합니다: **어떤 상황에 처한 사람**, 또는 **어떤 관점이나 수준의 독자**에게 특히 유익한지 설명해주세요.

    3. 3권의 책을 어떤 순서로 읽는 것이 가장 효과적인지 간단하게 제안해주세요. 
    - **이해도, 난이도, 주제의 흐름**을 고려해 순서를 정하고, 그 이유도 덧붙여 설명해주세요.

    ※ 응답은 아래 형식에 맞춰 정돈된 문장으로 작성해주세요:

    1. [도서 제목] (저자)
    - 추천 이유: 
    - 주요 내용:
    - 이런 분께 추천합니다:

    2. [도서 제목] (저자)
    - 추천 이유: 
    - 주요 내용:
    - 이런 분께 추천합니다:

    3. [도서 제목] (저자)
    - 추천 이유: 
    - 주요 내용:
    - 이런 분께 추천합니다:

    [읽기 순서 추천]
    - 1 → 2 → 3 순으로 읽기를 권장합니다. 이유: (간단한 설명 추가)
    """
    return prompt

def call_openai_api(prompt, model="gpt-4o"):
    """
    OpenAI API를 호출하여 추천 결과 받기
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "당신은 독서를 좋아하는 사람들에게 책을 추천해주는 전문가입니다. 주어진 도서 정보를 바탕으로 사용자의 요청에 가장 적합한 책을 추천해주세요."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API 호출 중 오류 발생: {e}"

def get_enhanced_user_profile(request_text):
    """
    사용자 요청에서 핵심 정보를 추출하는 함수
    연령대, 관심사, 독서 목적, 독서 수준을 추출합니다.
    """
    user_profile = {
        'age_group': '전체',
        'interests': [],
        'reading_purpose': '일반',
        'reading_level': '중급'
    }
    
    # 연령대 추출
    age_patterns = {
        '유아': ['유아', '아기', '유치원', '미취학', '0-7세', '영아'],
        '초등': ['초등', '어린이', '8-13세', '아동'],
        '청소년': ['청소년', '중학생', '고등학생', '틴에이저', '13-19세'],
        '성인': ['성인', '대학생', '20대', '30대', '40대', '직장인']
    }
    
    for age_group, keywords in age_patterns.items():
        if any(keyword in request_text.lower() for keyword in keywords):
            user_profile['age_group'] = age_group
            break
    
    # 관심사 추출
    interest_patterns = {
        '소설': ['소설', '이야기', '문학', '픽션', '판타지', '로맨스'],
        '역사': ['역사', '한국사', '세계사', '인물', '전기'],
        '과학': ['과학', '물리', '화학', '생물', '기술', 'STEM'],
        '자기계발': ['자기계발', '성공', '습관', '목표', '자기관리'],
        '어린이': ['그림책', '동화', '우리아이', '애니메이션'],
        '경제/경영': ['경제', '경영', '투자', '부동산', '주식', '창업'],
        '예술': ['예술', '음악', '미술', '디자인', '공예'],
        '교육': ['교육', '학습', '문제집', '참고서', '시험', '공부']
    }
    
    for category, keywords in interest_patterns.items():
        if any(keyword in request_text.lower() for keyword in keywords):
            user_profile['interests'].append(category)
    
    # 독서 목적 추출
    purpose_patterns = {
        '학습': ['공부', '학습', '배움', '스터디', '연구', '전공', '학업', '시험', '준비'],
        '취미': ['취미', '재미', '즐거움', '여가', '흥미', '관심사'],
        '자기계발': ['자기계발', '성장', '발전', '역량', '스킬', '능력'],
        '정보수집': ['정보', '지식', '업데이트', '트렌드', '소식', '현황'],
        '문제해결': ['문제', '해결', '도움', '방법', '팁', '전략'],
        '독후감': ['독후감', '과제', '숙제', '학교', '추천받기']
    }
    
    for purpose, keywords in purpose_patterns.items():
        if any(keyword in request_text.lower() for keyword in keywords):
            user_profile['reading_purpose'] = purpose
            break
    
    # 독서 수준 추출
    if any(word in request_text.lower() for word in ['입문', '처음', '초보', '기초', '쉬운', '시작']):
        user_profile['reading_level'] = '초급'
    elif any(word in request_text.lower() for word in ['심화', '전문', '고급', '깊이', '상세한', '어려운']):
        user_profile['reading_level'] = '고급'
    elif any(word in request_text.lower() for word in ['중급', '적당한', '중간']):
        user_profile['reading_level'] = '중급'
    
    return user_profile

def run_hybrid_recommendation(recommendation_system, user_request, model="gpt-4o", num_candidates=15):
    """
    하이브리드 도서 추천 실행 함수 (ML + LLM)
    """
    # 사용자 요청에서 프로필 정보 추출 (향상된 버전 사용)
    user_profile = get_enhanced_user_profile(user_request)
    
    # ML 기반 후보 도서 선택 (15개)
    top_candidates = select_books_for_recommendation(
        user_profile, 
        recommendation_system['data'], 
        n=num_candidates
    )
    
    # LLM 프롬프트 생성
    prompt = create_recommendation_prompt(user_request, top_candidates)
    
    # OpenAI API 호출
    recommendation_result = call_openai_api(prompt, model)
    
    # 결과 반환
    return {
        "prompt": prompt,
        "recommendation": recommendation_result,
        "user_profile": user_profile,
        "top_candidates": top_candidates
    }

# -------------------------------
# Streamlit 웹 앱
# -------------------------------
st.set_page_config(page_title="공공도서관 하이브리드 도서 추천 시스템", layout="wide")
st.title("📚 북큐레이터(BookCurator): AI 기반 맞춤형 도서 추천 시스템")

st.markdown("""
북큐레이터(BookCurator)는 머신러닝과 LLM 모델을 결합한 하이브리드 방식으로 도서를 추천합니다.
1. 머신러닝으로 사용자 프로필에 맞는 도서 후보군 선정
2. LLM 모델로 최종 추천 및 상세한 추천 이유 생성
""")

# 사이드바 설정
st.sidebar.title("⚙️ 설정")
data_path = st.sidebar.text_input("데이터 파일 경로:", "loan_information_processed.csv")
model_option = st.sidebar.selectbox(
    "LLM 모델 선택:",
    ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
    index=2
)
num_candidates = st.sidebar.slider(
    "ML 추천 후보 수:",
    min_value=5,
    max_value=20,
    value=15,
    step=1,
    help="머신러닝으로 선정할 도서 후보 수"
)

# 데이터 및 추천 시스템 로드
try:
    if 'recommendation_system' not in st.session_state:
        with st.spinner("추천 시스템 로드 중..."):
            st.session_state.recommendation_system = load_recommendation_system(data_path)
    
    recommendation_system = st.session_state.recommendation_system
    df = recommendation_system['data']
    
    st.sidebar.success(f"추천 시스템 로드 완료: {len(df)} 권의 도서 정보")
    
    # 데이터 통계 표시
    with st.sidebar.expander("📊 데이터 통계"):
        st.write(f"총 도서 수: {len(df)}")
        st.write(f"연령대 구분: {df['age_group'].nunique()}")
        st.write(f"카테고리 구분: {df['estimated_category'].nunique()}")
        st.write(f"출판사 수: {df['publisher'].nunique()}")
        
        # 인기 카테고리 표시
        st.write("인기 카테고리:")
        category_counts = df['estimated_category'].value_counts().head(5)
        for category, count in category_counts.items():
            st.write(f"- {category}: {count}권")
    
    # 사용자 입력
    user_input = st.text_area(
        "📨 추천 요청 문장을 입력하세요:", 
        "중학생 딸이 과학에 관심을 보이기 시작했어요. 청소년이 쉽게 읽을 수 있는 과학책 추천해주세요.",
        height=100
    )
    
    # 추가 정보 입력
    with st.expander("🔍 추가 정보 입력 (선택사항)"):
        col1, col2 = st.columns(2)
        with col1:
            selected_age = st.selectbox(
                "연령대:",
                ["자동 감지", "유아", "초등", "청소년", "성인", "전체"],
                index=0
            )
        with col2:
            selected_purpose = st.selectbox(
                "독서 목적:",
                ["자동 감지", "학습", "취미", "자기계발", "정보수집", "문제해결", "독후감", "일반"],
                index=0
            )
        
        selected_interests = st.multiselect(
            "관심사:",
            ["소설", "역사", "과학", "자기계발", "어린이", "경제/경영", "예술", "교육"],
            default=[]
        )
        
        selected_level = st.radio(
            "독서 수준:",
            ["자동 감지", "초급", "중급", "고급"],
            index=0,
            horizontal=True
        )

    if st.button("🤖 하이브리드 추천 받기 (ML + LLM)", use_container_width=True):
        with st.spinner(f"하이브리드 추천 생성 중 😁 (30초 이상 소요될 수 있습니다.)"):
            try:
                # 사용자 프로필 추출
                user_profile = get_enhanced_user_profile(user_input)
                
                # 추가 정보 입력이 있으면 사용
                if selected_age != "자동 감지":
                    user_profile['age_group'] = "전체" if selected_age == "전체" else selected_age
                
                if selected_purpose != "자동 감지":
                    user_profile['reading_purpose'] = selected_purpose
                
                if selected_interests:
                    user_profile['interests'] = selected_interests
                
                if selected_level != "자동 감지":
                    user_profile['reading_level'] = selected_level
                
                # 하이브리드 추천 실행
                results = run_hybrid_recommendation(
                    recommendation_system,
                    user_input,
                    model_option,
                    num_candidates
                )
                
                # 결과 표시
                st.markdown("### 📋 추출된 사용자 프로필")
                st.json(results["user_profile"])
                
                st.markdown("### 🤖 LLM 추천 결과")
                st.markdown(results["recommendation"])
                
                # ML 후보 정보
                with st.expander(f"ML 후보 도서 정보 ({len(results['top_candidates'])}권)"):
                    candidate_df = results['top_candidates'][['title', 'main_author', 'publisher', 'estimated_category', 'age_group', 'total_score']].copy()
                    candidate_df['total_score'] = candidate_df['total_score'].apply(lambda x: f"{x:.2f}")
                    candidate_df.columns = ['제목', '저자', '출판사', '카테고리', '대상 독자', '적합도 점수']
                    st.dataframe(candidate_df, use_container_width=True)
                
                # 프롬프트 표시
                with st.expander("LLM에 전송된 프롬프트 보기"):
                    st.code(results["prompt"], language="text")
            
            except Exception as e:
                st.error(f"하이브리드 추천 생성 중 오류 발생: {e}")
    
    # 시스템 소개
    with st.expander("📌 하이브리드 추천 시스템 정보"):
        st.markdown("""
        ### 🔍 하이브리드 도서 추천 시스템 작동 방식
        
        북큐레이터(BookCurator)는 머신러닝과 LLM를 결합한 하이브리드 방식으로 도서를 추천합니다.
        
        **1️⃣단계: 사용자 프로필 엔티티 추출**
        - 자연어 요청에서 연령대, 관심사, 독서 목적, 독서 수준 추출
        - 사용자가 직접 입력한 추가 정보가 있으면 반영
        
        **2️⃣단계: ML 기반 후보 도서 선정**
        - 사용자 프로필에 맞는 도서 필터링
        - 연령대, 관심사, 인기도 등을 고려한 점수 계산
        - 다양한 저자와 출판사의 도서를 포함하도록 최적화
        
        **3️⃣단계: LLM 기반 최종 추천**
        - ML이 선정한 후보 도서들을 LLM에 전달
        - 사용자 요청과 도서 정보를 바탕으로 최적의 3권 선별
        - 각 도서에 대한 상세한 추천 이유와 설명 생성
        
        **📌 하이브리드 방식의 장점**
        - ML: 대량의 도서 중에서 효율적으로 적합한 후보 선별
        - LLM: 도서 내용을 이해하고 사용자 맥락에 맞는 상세한 추천 이유 제공
        - 두 기술의 장점을 결합하여 더 정확하고 유용한 추천 제공
        """)

except Exception as e:
    st.error(f"시스템 초기화 중 오류 발생: {e}")
    st.info(f"데이터 파일 경로를 확인해주세요: {data_path}")