import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

def preprocess_book_data(csv_path):
    """
    도서 데이터 전처리 함수
    """
    print("1. 데이터 로딩 중...")
    df = pd.read_csv(csv_path)
    print(f"데이터 크기: {df.shape}")
    
    # 컬럼명 영어로 변경 (작업 편의성)
    column_mapping = {
        'RANK_CO': 'rank',
        'BOOK_TITLE_NM': 'title',
        'AUTHR_NM': 'author',
        'BOOK_INTRCN_CN': 'description',
        'PUBLISHER_NM': 'publisher',
        'AGE_FLAG_NM': 'age_group',
        'SEXDSTN_FLAG_NM': 'gender'
    }
    df = df.rename(columns=column_mapping)
    
    # 결측치 처리
    # 필수 정보인 제목, 저자, 출판사만 있으면 다른 정보는 없어도 됨
    df_clean = df.dropna(subset=['title', 'author', 'publisher'])
    
    # 나머지 컬럼 결측치 처리
    df_clean['description'] = df_clean['description'].fillna('설명 없음')
    df_clean['age_group'] = df_clean['age_group'].fillna('전체')
    df_clean['gender'] = df_clean['gender'].fillna('전체')
    
    # 순위 데이터 정수형으로 변환
    df_clean['rank'] = pd.to_numeric(df_clean['rank'], errors='coerce').fillna(999).astype(int)
    
    # 텍스트 데이터 정제
    df_clean['clean_description'] = df_clean['description'].apply(clean_text)
    
    # 설명 길이 특성 추가
    df_clean['description_length'] = df_clean['clean_description'].apply(len)
    
    # 저자 정보 정제
    df_clean['main_author'] = df_clean['author'].apply(extract_main_author)
    
    # 출판사 정보 정제
    df_clean['clean_publisher'] = df_clean['publisher'].str.strip()
    
    # 시리즈 정보 추출
    df_clean['is_series'] = df_clean['clean_description'].apply(extract_series)
    
    # 대상 독자층 추정
    df_clean['estimated_audience'] = df_clean['clean_description'].apply(estimate_target_audience)
    
    # 도서 장르/카테고리 추정
    df_clean['estimated_category'] = df_clean['clean_description'].apply(estimate_category)
    
    # 연령대 추정 강화
    df_clean['age_group'] = df_clean.apply(estimate_age_group, axis=1)
    
    print("데이터 전처리 완료!")
    return df_clean

def clean_text(text):
    """텍스트 데이터 정제 함수"""
    if isinstance(text, str):
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        # 특수문자 제거
        text = re.sub(r'[^\w\s]', ' ', text)
        # 여러 공백을 하나로 변경
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ''

def extract_main_author(author_text):
    """저자 정보에서 주 저자 추출"""
    if not isinstance(author_text, str):
        return ''
    
    # '글:', '지은이:', '지음' 등의 표현과 이후 내용 추출
    patterns = [r'글\s*:\s*([^;]+)', r'지은이\s*:\s*([^;]+)', r'([^;]+)\s*지음', r'^([^;]+)']
    
    for pattern in patterns:
        match = re.search(pattern, author_text)
        if match:
            return match.group(1).strip()
    
    # 기본적으로 첫 번째 저자만 반환
    if ';' in author_text:
        return author_text.split(';')[0].strip()
    elif ',' in author_text:
        return author_text.split(',')[0].strip()
    
    return author_text.strip()

def extract_series(text):
    """시리즈 정보 추출"""
    if not isinstance(text, str):
        return np.nan
    
    series_patterns = [
        r'시리즈\s*(\d+)권', 
        r'(\w+)\s*시리즈', 
        r'(\d+)권'
    ]
    
    for pattern in series_patterns:
        match = re.search(pattern, text)
        if match:
            return True
    
    return False

def estimate_target_audience(text):
    """대상 독자층 추정"""
    if not isinstance(text, str):
        return '알 수 없음'
    
    text = text.lower()
    
    if any(word in text for word in ['어린이', '아동', '유아', '초등']):
        return '어린이'
    elif any(word in text for word in ['청소년', '중학생', '고등학생']):
        return '청소년'
    elif any(word in text for word in ['대학', '전문가', '성인']):
        return '성인'
    else:
        return '일반'

def estimate_category(text):
    """도서 장르/카테고리 추정"""
    if not isinstance(text, str):
        return '알 수 없음'
    
    text = text.lower()
    
    categories = {
        '소설': ['소설', '이야기', '문학', '픽션'],
        '역사': ['역사', '한국사', '세계사'],
        '과학': ['과학', '물리', '화학', '생물', '기술'],
        '자기계발': ['자기계발', '성공', '습관', '목표'],
        '어린이': ['그림책', '동화', '우리아이'],
        '경제/경영': ['경제', '경영', '투자', '부동산', '주식'],
        '예술': ['예술', '음악', '미술', '디자인'],
        '교육': ['교육', '학습', '문제집', '참고서']
    }
    
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category
    
    return '기타'

def estimate_age_group(row):
    """연령대 추정 강화"""
    if pd.notna(row['age_group']) and row['age_group'] != '전체':
        return row['age_group']
    
    # 텍스트에서 연령 힌트 찾기
    text = str(row['title']) + ' ' + str(row['description'])
    text = text.lower()
    
    if any(term in text for term in ['유아', '아기', '유치원', '0-7세', '영아']):
        return '유아(0~7)'
    elif any(term in text for term in ['초등', '어린이', '8-13세', '동화']):
        return '초등(8~13)'
    elif any(term in text for term in ['청소년', '중학생', '고등학생', '13-19세']):
        return '청소년(13~19)'
    elif any(term in text for term in ['성인', '20대', '30대', '대학생']):
        return '성인(20대~)'
    
    # 출판사 기반 추정 - 어린이 도서 전문 출판사 리스트
    children_publishers = ['보림', '길벗어린이', '사계절', '비룡소', '국민서관', '창비', '웅진주니어', '한솔수북']
    if row['clean_publisher'] in children_publishers:
        return '초등(8~13)'
    
    return '전체'

if __name__ == "__main__":
    # CSV 파일 경로
    csv_path = './loan_information_no_duplicates.csv'
    
    # 데이터 전처리
    processed_df = preprocess_book_data(csv_path)
    
    # 결과 저장
    processed_df.to_csv('loan_information_processed.csv', index=False)
    print("처리된 데이터가 'loan_information_processed.csv'에 저장되었습니다.")
