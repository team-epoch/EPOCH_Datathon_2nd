import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def build_recommendation_system(processed_data_path):
    """
    머신러닝 기반 도서 추천 시스템 구축 함수
    """
    print("머신러닝 및 추천 시스템 구현 중")
    
    # 전처리된 데이터 로드
    df_clean = pd.read_csv(processed_data_path)
    
    # complexity 컬럼이 있으면 제거
    if 'complexity' in df_clean.columns:
        df_clean = df_clean.drop('complexity', axis=1)
    
    # 추천 시스템을 위한 특성 추출
    print("1. TF-IDF 특성 추출 중")
    tfidf_features = create_tfidf_features(df_clean)
    
    # 장르 예측 모델 학습
    print("2. 장르 예측 모델 구축 중")
    genre_prediction_model = train_genre_prediction_model(df_clean)
    
    # 추천 시스템 관련 데이터 준비
    print("3. 추천 시스템 데이터 준비 중")
    
    # 연령대-성별 그룹별 인기 도서
    popularity_by_group = df_clean.groupby(['age_group', 'gender'])['title'].agg(
        popularity_count='count'
    ).reset_index().sort_values('popularity_count', ascending=False)
    
    # 연령대별 인기 출판사
    publisher_popularity = df_clean.groupby(['age_group', 'publisher']).size().reset_index(name='count')
    top_publishers = publisher_popularity.sort_values('count', ascending=False).groupby('age_group').head(5)
    
    # 추천 시스템 관련 함수 및 데이터 반환
    recommendation_system = {
        'data': df_clean,
        'tfidf_features': tfidf_features,
        'genre_model': genre_prediction_model,
        'popularity_by_group': popularity_by_group,
        'top_publishers': top_publishers
    }
    
    print("추천 시스템 구축 완료!")
    return recommendation_system

def create_tfidf_features(df):
    """TF-IDF 벡터화 (설명 텍스트 기반)"""
    try:
        tfidf = TfidfVectorizer(max_features=1000, stop_words=['등', '및', '을', '를', '이', '가', '의', '에'])
        tfidf_matrix = tfidf.fit_transform(df['clean_description'].fillna(''))
        return {
            'vectorizer': tfidf,
            'matrix': tfidf_matrix
        }
    except Exception as e:
        print(f"TF-IDF 특성 추출 중 오류 발생: {e}")
        return None

def train_genre_prediction_model(df):
    """장르 예측 개선 (머신러닝 기반)"""
    try:
        # 학습 데이터 생성 (기존 추정 카테고리 중 '기타'가 아닌 것만 선별)
        labeled_data = df[df['estimated_category'] != '기타'].sample(
            min(1000, len(df[df['estimated_category'] != '기타'])),
            random_state=42
        )
        
        # 텍스트 특성 추출
        mini_tfidf = TfidfVectorizer(max_features=500)
        X = mini_tfidf.fit_transform(labeled_data['clean_description'])
        y = labeled_data['estimated_category']
        
        # 분류 모델 학습
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 미분류 데이터에 적용
        unlabeled_idx = df[df['estimated_category'] == '기타'].index
        if len(unlabeled_idx) > 0:
            X_unlabeled = mini_tfidf.transform(df.loc[unlabeled_idx, 'clean_description'])
            df.loc[unlabeled_idx, 'estimated_category'] = model.predict(X_unlabeled)
        
        return {
            'model': model,
            'vectorizer': mini_tfidf
        }
    except Exception as e:
        print(f"장르 예측 모델 학습 중 오류 발생: {e}")
        return None

def get_candidates_for_user(user_profile, df, n=15):
    """
    사용자 프로필에 맞는 후보 도서 추출
    
    Args:
        user_profile: 사용자 정보 딕셔너리 (연령대, 관심사 등)
        df: 전체 도서 데이터프레임
        n: 반환할 후보 수
    
    Returns:
        후보 도서 데이터프레임
    """
    filtered_df = df.copy()
    
    # 연령대 필터링
    if 'age_group' in user_profile and user_profile['age_group'] != '전체':
        filtered_df = filtered_df[filtered_df['age_group'] == user_profile['age_group']]
    
    # 카테고리 필터링
    if 'interests' in user_profile and user_profile['interests']:
        filtered_df = filtered_df[filtered_df['estimated_category'].isin(user_profile['interests'])]
    
    # 후보가 너무 적으면 필터 완화
    if len(filtered_df) < n:
        filtered_df = df.copy()
        if 'age_group' in user_profile and user_profile['age_group'] != '전체':
            filtered_df = filtered_df[filtered_df['age_group'] == user_profile['age_group']]
    
    # 여전히 후보가 부족하면 전체에서 선택
    if len(filtered_df) < n:
        filtered_df = df.copy()
    
    # 랭크가 높은(낮은 숫자) 책들 우선 선택
    return filtered_df.sort_values('rank').head(n)

def select_books_for_recommendation(user_profile, df, n=15):
    """
    사용자 프로필에 맞게 추천할 도서 선택
    
    Args:
        user_profile: 사용자 프로필 딕셔너리
        df: 전체 도서 데이터프레임
        n: 반환할 추천 도서 수
        
    Returns:
        추천 도서 데이터프레임
    """
    # 1단계: 기본 필터링
    candidates = get_candidates_for_user(user_profile, df, n=min(n*5, 100))
    
    # 2단계: 스코어링
    candidates = candidates.copy()
    
    # 연령대 적합성 점수
    if user_profile['age_group'] != '전체':
        candidates['age_score'] = candidates['age_group'].apply(
            lambda x: 1.0 if x == user_profile['age_group'] else 0.2
        )
    else:
        candidates['age_score'] = 1.0
        
    # 관심사 적합성 점수
    if user_profile['interests']:
        candidates['interest_score'] = candidates['estimated_category'].apply(
            lambda x: 1.0 if x in user_profile['interests'] else 0.2
        )
    else:
        candidates['interest_score'] = 1.0
        
    # 인기도 점수 (순위가 낮을수록 인기)
    max_rank = candidates['rank'].max()
    candidates['popularity_score'] = 1 - (candidates['rank'] / max_rank)
    
    # 총점 계산 (복잡성 점수 제외)
    candidates['total_score'] = (
        candidates['age_score'] * 0.4 +
        candidates['interest_score'] * 0.4 +
        candidates['popularity_score'] * 0.2
    )
    
    # 3단계: 다양성 보장
    top_books = candidates.sort_values('total_score', ascending=False).head(n)
    
    # 다양한 저자와 출판사 보장
    if len(top_books) < n and len(candidates) > n:
        # 이미 선택된 저자와 출판사
        selected_authors = set(top_books['main_author'])
        selected_publishers = set(top_books['clean_publisher'])
        
        # 나머지 후보들
        remaining = candidates[~candidates.index.isin(top_books.index)]
        
        # 다양성을 위한 추가 선택
        while len(top_books) < n and not remaining.empty:
            # 다른 저자/출판사의 책 우선 선택
            diversity_candidates = remaining[
                (~remaining['main_author'].isin(selected_authors)) | 
                (~remaining['clean_publisher'].isin(selected_publishers))
            ]
            
            if diversity_candidates.empty:
                diversity_candidates = remaining
                
            # 점수가 가장 높은 책 추가
            next_book = diversity_candidates.sort_values('total_score', ascending=False).iloc[0]
            top_books = pd.concat([top_books, pd.DataFrame([next_book])])
            
            # 선택한 책 제거 및 저자/출판사 업데이트
            remaining = remaining[remaining.index != next_book.name]
            selected_authors.add(next_book['main_author'])
            selected_publishers.add(next_book['clean_publisher'])
    
    return top_books.head(n)

def parse_user_request(request_text):
    """
    사용자 요청에서 핵심 정보를 추출하는 함수
    
    Args:
        request_text: 사용자 요청 텍스트
        
    Returns:
        사용자 프로필 딕셔너리
    """
    user_profile = {
        'age_group': '전체',
        'interests': []
    }
    
    # 연령대 추출
    age_patterns = {
        '유아(0~7)': ['유아', '아기', '유치원', '미취학', '0-7세', '영아'],
        '초등(8~13)': ['초등', '어린이', '8-13세', '아동'],
        '청소년(13~19)': ['청소년', '중학생', '고등학생', '틴에이저', '13-19세'],
        '성인(20대~)': ['성인', '대학생', '20대', '30대', '40대', '직장인']
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
    
    return user_profile

if __name__ == "__main__":
    # 전처리된 데이터 파일 경로
    processed_data_path = './loan_information_processed.csv'
    
    # 추천 시스템 구축
    recommendation_system = build_recommendation_system(processed_data_path)
    
    # 추천 시스템 관련 데이터 저장
    np.save('tfidf_matrix.npy', recommendation_system['tfidf_features']['matrix'].toarray())
    
    # 사용자 요청 테스트
    test_request = "초등학생 아이가 과학에 관심이 많은데 재미있게 읽을 수 있는 책을 추천해주세요."
    user_profile = parse_user_request(test_request)
    print(f"추출된 사용자 프로필: {user_profile}")
    
    # 추천 도서 선택
    recommended_books = select_books_for_recommendation(
        user_profile, 
        recommendation_system['data'], 
        n=15
    )
    
    print(f"추천 도서 수: {len(recommended_books)}")
    print("추천 도서 목록:")
    for _, book in recommended_books.iterrows():
        print(f"- {book['title']} (저자: {book['main_author']})")