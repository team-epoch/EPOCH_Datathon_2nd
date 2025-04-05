import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from preprocess import get_data

def recommend_books(df, num_clusters, top_percentage_range=(0.2, 0.5), num_recommendations=5):
    recommendations = {}

    for cluster_id in range(num_clusters):
        print(f"\n클러스터 {cluster_id}에 대한 책 추천:")

        # 클러스터에 속한 데이터 추출
        cluster_books = df[df['cluster'] == cluster_id]

        sorted_cluster_books = cluster_books.sort_values(by='ANALS_LON_TOT_CO', ascending=False)

        total_books = len(sorted_cluster_books)
        lower_bound = int(total_books * top_percentage_range[0])
        upper_bound = int(total_books * top_percentage_range[1])

        selected_books = sorted_cluster_books.iloc[lower_bound:upper_bound]

        random_books = selected_books.sample(n=min(num_recommendations, len(selected_books)))
        recommendations[cluster_id] = random_books[['BOOK_TITLE_NM', 'AUTHR_NM', 'ANALS_LON_TOT_CO']]
        print(random_books[['BOOK_TITLE_NM', 'AUTHR_NM', 'ANALS_LON_TOT_CO']])

    return recommendations

loan_info = pd.read_csv(
    r'C:\Users\ksost\OneDrive\바탕 화면\냥대\에포크\mini_project\data\loan_information\loan_information.csv',
    low_memory=False
)
library_books = pd.read_csv(
    r'C:\Users\ksost\OneDrive\바탕 화면\냥대\에포크\mini_project\data\books\books_2.csv',
    low_memory=False
)

df = get_data(loan_info, library_books)
df.columns

combined_features = np.load('combined_features.npy')
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(combined_features)

score = silhouette_score(combined_features, df['cluster'])
print(f"\n 실루엣 계수 (Silhouette Score): {score:.4f}")

recommended_books = recommend_books(df, num_clusters=5, top_percentage_range=(0.2, 0.5), num_recommendations=5)

for cluster_id, books in recommended_books.items():
    print(f"\n추천된 책들 - 클러스터 {cluster_id}: ", books)