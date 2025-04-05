import pandas as pd

def get_data(loan_info, library_books):
    # 2. loan_info에서 결측치 있는 책만 추출
    loan_missing = loan_info[loan_info['BOOK_INTRCN_CN'].isna()].copy()

    # 3. library_books에서 TITLE_NM, BOOK_INTRCN_CN 추출
    lib_info = library_books[['TITLE_NM', 'BOOK_INTRCN_CN']].dropna().drop_duplicates(subset='TITLE_NM')

    # 4. loan_missing의 BOOK_TITLE_NM 컬럼을 TITLE_NM으로 rename해서 병합
    loan_missing = pd.merge(
        loan_missing,
        lib_info,
        left_on='BOOK_TITLE_NM',
        right_on='TITLE_NM',
        how='left'
    )

    # 5. 결측치 채우기: loan_info의 BOOK_INTRCN_CN이 결측일 경우, library_books의 값으로 채움
    loan_missing['BOOK_INTRCN_CN'] = loan_missing['BOOK_INTRCN_CN_y'].combine_first(loan_missing['BOOK_INTRCN_CN_x'])

    # 6. 정리: 원래 loan_info 형식으로 복원
    loan_missing_cleaned = loan_missing[loan_info.columns]  # 컬럼 순서 맞추기

    # 7. 기존 소개 있는 행 + 채워진 행 합치기
    loan_info_filled = pd.concat([
        loan_info[loan_info['BOOK_INTRCN_CN'].notna()],
        loan_missing_cleaned
    ], ignore_index=True)

    # 8. 최종 중복 제거 (혹시 중복된 BOOK_TITLE_NM이 있다면)
    loan_info_final = loan_info_filled.drop_duplicates(subset='BOOK_TITLE_NM', keep='first')
    
    # 9. 결과 확인
    print("최종 loan_info shape:", loan_info_final.shape)
    print("아직 소개글 없는 책 수:", loan_info_final['BOOK_INTRCN_CN'].isna().sum())

    genre_info = library_books[['TITLE_NM', 'READER', 'PUBLISH_TYPE', 'CATE']].drop_duplicates(subset='TITLE_NM')
    genre_info.rename(columns={'TITLE_NM': 'BOOK_TITLE_NM'}, inplace=True)

    loan_info_with_codes = pd.merge(
        loan_info_final,
        genre_info,
        on='BOOK_TITLE_NM',
        how='left'  # loan_info 기준으로 병합
    )

    loan_info_with_codes[['READER', 'PUBLISH_TYPE', 'CATE']] = loan_info_with_codes[[
    'READER', 'PUBLISH_TYPE', 'CATE']].fillna(0).astype(int)
    
    data = loan_info_with_codes
    print(data.columns)
    print(data.head())

    return data

# def merge_genre(df, library_books):
#     genre_info = library_books[['TITLE_NM', 'REDER', 'PUBLISH_TYPE', 'CATE']].drop_duplicates(subset='TITLE_NM')
#     genre_info.rename(columns={'TITLE_NM': 'BOOK_TITLE_NM'}, inplace=True)

#     loan_info_with_codes = pd.merge(
#         df,
#         genre_info,
#         on='BOOK_TITLE_NM',
#         how='left'  # loan_info 기준으로 병합
#     )
#     return loan_info_with_codes