import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from preprocess import get_data
import joblib

loan_info = pd.read_csv(
    r'C:\Users\ksost\OneDrive\바탕 화면\냥대\에포크\mini_project\data\loan_information\loan_information.csv',
    low_memory=False
)
library_books = pd.read_csv(
    r'C:\Users\ksost\OneDrive\바탕 화면\냥대\에포크\mini_project\data\books\books_2.csv',
    low_memory=False
)

df = get_data(loan_info, library_books)

code_vector = df[['READER', 'PUBLISH_TYPE', 'CATE']].values

df['full_text'] = df.apply(lambda row: f"{row['BOOK_TITLE_NM']} [SEP] {row['AUTHR_NM']} [SEP] {row['BOOK_INTRCN_CN']}", axis=1)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(df['full_text'].tolist(), batch_size=128, show_progress_bar=True)

combined_features = np.hstack((embeddings, code_vector))
np.save('combined_features.npy', combined_features)

num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(combined_features)

joblib.dump(kmeans, 'kmeans_model.pkl') 

sse = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(combined_features)
    sse.append(kmeans.inertia_)

plt.plot(range(2, 15), sse, marker='o')
plt.title('Elbow Method - SSE vs k')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE (Inertia)')
plt.grid(True)
plt.show()


score = silhouette_score(combined_features, df['cluster'])
print(f"\n 실루엣 계수 (Silhouette Score): {score:.4f}")

for cluster_id in range(num_clusters):
    print(f"\n 클러스터 {cluster_id}")
    print(df[df['cluster'] == cluster_id][['BOOK_TITLE_NM', 'AUTHR_NM']].head(5))

pca = PCA(n_components=3)
points = pca.fit_transform(combined_features)

plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], c=df['cluster'], cmap='tab10')
plt.title("clustering results (PCA 2D)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.colorbar()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=3)
points_3d = pca.fit_transform(combined_features)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=df['cluster'], cmap='tab10', s=10)
ax.set_title("clustering results(PCA 3D)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_zlabel("PCA3")
plt.colorbar(scatter)
plt.show()
