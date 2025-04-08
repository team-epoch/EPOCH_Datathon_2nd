import pandas as pd
import numpy as np
from geopy.distance import geodesic
import re

def preprocess_library_data(csv_path):
    """
    도서관 데이터 전처리 함수
    
    Args:
        csv_path: 도서관 CSV 파일 경로
        
    Returns:
        전처리된 도서관 데이터프레임
    """
    # 도서관 데이터 로드
    lib_df = pd.read_csv(csv_path)
    
    # 필요한 컬럼만 선택
    useful_columns = [
        'LBRRY_CD', 'LBRRY_NM', 'LBRRY_ADDR', 
        'LBRRY_LA', 'LBRRY_LO', 'ONE_AREA_NM', 
        'TWO_AREA_NM', 'TEL_NO', 'HMPG_VALUE', 
        'OPNNG_TIME', 'CLOSEDON_DC'
    ]
    
    lib_df = lib_df[useful_columns]
    
    # 컬럼명 변경
    column_mapping = {
        'LBRRY_CD': 'library_id',
        'LBRRY_NM': 'library_name',
        'LBRRY_ADDR': 'address',
        'LBRRY_LA': 'latitude',
        'LBRRY_LO': 'longitude',
        'ONE_AREA_NM': 'province',
        'TWO_AREA_NM': 'city',
        'TEL_NO': 'phone',
        'HMPG_VALUE': 'website',
        'OPNNG_TIME': 'opening_hours',
        'CLOSEDON_DC': 'closed_days'
    }
    
    lib_df = lib_df.rename(columns=column_mapping)
    
    # 위도, 경도를 float 타입으로 변환
    lib_df['latitude'] = pd.to_numeric(lib_df['latitude'], errors='coerce')
    lib_df['longitude'] = pd.to_numeric(lib_df['longitude'], errors='coerce')
    
    # 결측치 처리
    lib_df = lib_df.dropna(subset=['latitude', 'longitude', 'library_name', 'address'])
    
    # 지역 정보 통합 (검색 용이성을 위해)
    lib_df['location'] = lib_df['province'] + ' ' + lib_df['city']
    
    return lib_df

def extract_location_from_request(request_text):
    """
    사용자 요청에서 위치 정보 추출
    
    Args:
        request_text: 사용자 요청 텍스트
        
    Returns:
        추출된 위치 정보 (없으면 None)
    """
    # 주요 지역명 패턴
    location_patterns = [
        r'(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)(?:특별시|광역시|특별자치시|도|특별자치도)?',
        r'((?:강남|강동|강북|강서|관악|광진|구로|금천|노원|도봉|동대문|동작|마포|서대문|서초|성동|성북|송파|양천|영등포|용산|은평|종로|중구|중랑)구)',
        r'(.+시 .+구)',
        r'(.+[시군])'
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, request_text)
        if match:
            return match.group(0)
    
    return None

def find_nearby_libraries(user_location, library_df, max_count=3):
    """
    사용자 위치와 가까운 도서관 찾기
    
    Args:
        user_location: 사용자 위치 (지역명)
        library_df: 도서관 데이터프레임
        max_count: 반환할 최대 도서관 수
        
    Returns:
        가까운 도서관 정보 데이터프레임
    """
    # 지역명 포함 여부로 필터링
    matching_libs = library_df[
        library_df['location'].str.contains(user_location) | 
        library_df['address'].str.contains(user_location)
    ]
    
    if len(matching_libs) == 0:
        return None
    
    # 최대 max_count개 반환
    return matching_libs.head(max_count)

def format_library_info(libraries_df):
    """
    도서관 정보를 포맷팅
    """
    if libraries_df is None or len(libraries_df) == 0:
        return "해당 지역의 도서관 정보를 찾을 수 없습니다."
    
    info = "🏛️ 인근 도서관 정보:\n\n"
    
    for _, lib in libraries_df.iterrows():
        info += f"📚 **{lib['library_name']}**\n"
        info += f"- 주소: {lib['address']}\n"
        info += f"- 전화: {lib['phone']}\n"
        info += f"- 운영시간: {lib['opening_hours']}\n"
        info += f"- 휴관일: {lib['closed_days']}\n\n"
    
    return info