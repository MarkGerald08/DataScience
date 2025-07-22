import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt.rcParams["figure.dpi"] = 300

# ================================================================
# Data loading
# ================================================================
df = pd.read_csv("../data/processed/gender_ineq_data.csv")
df.head()


# ================================================================
# Data Viz
# ================================================================
# Visualizing data distribution
df.plot(kind="box", figsize=(12, 6), rot=50, title="Feature Distribution")
plt.tight_layout()
plt.show()

# Visualizing correlation
corr_mat = df.corr(numeric_only=True)

plt.figure(figsize=(12, 9))
sns.heatmap(corr_mat, annot=True, robust=True, cbar=False)
plt.tight_layout()
plt.show()

# Visualizing data relation
plt.scatter(data=df, x="Secondary Education(f)", y="Secondary Education(m)")
plt.xlabel("Secondary Education(f)")
plt.ylabel("Secondary Education(m)")
plt.tight_layout()
plt.show()

plt.scatter(data=df, x="Labour Force(f)", y="Labour Force(m)")
plt.xlabel("Labor Force(f)")
plt.ylabel("Labor Force(m)")
plt.tight_layout()
plt.show()

# ================================================================
# KMeans Clustering
# ================================================================
# Clustering for labor force.
lf_cols = ["Labour Force(f)", "Labour Force(m)"]

scaler = StandardScaler()
scaled_lf = scaler.fit_transform(df[lf_cols])

# Elbow method using inertia
"""Elbow method to identify the optimal value of k.
    """
inertia = []
k_value = range(1, 7)
for k in k_value:
    kmeans = KMeans(random_state=42, n_clusters=k)
    kmeans.fit(scaled_lf)
    inertia.append(kmeans.inertia_)

plt.plot(k_value, inertia)
plt.axis([1, 6, 66, 390])
plt.title("Inertia per Number of Clusters")
plt.tight_layout()
plt.show()

fig = px.line(x=k_value, y=inertia)
fig.update_layout(
    title="Inertia per Cluster",
    xaxis_title="k",
    yaxis_title="Inertia"
)
fig.show()

# Silhouette score for labor force
sil_score = []
k_value = range(2, 7)
for k in k_value:
    kmeans = KMeans(random_state=42, n_clusters=k)
    kmeans.fit(scaled_lf)
    sil_score.append(silhouette_score(scaled_lf, kmeans.labels_))

plt.plot(k_value, sil_score)
plt.axis([2, 6, 0.34, 0.42])
plt.title("Silhouetter Score per Number of Clusters")
plt.tight_layout()
plt.show()

fig = px.line(x=k_value, y=sil_score)
fig.update_layout(
    title="Score per Cluster",
    xaxis_title="k",
    yaxis_title="Silhouette Score"
)
fig.show()

# Applying the optimal value.
kmeans = KMeans(n_clusters=3, random_state=0)
df["lf_cluster"] = kmeans.fit_predict(df[lf_cols])


# Clustering for Secondary Education.
educ_cols = ["Secondary Education(f)", "Secondary Education(m)"]

scaled_educ = scaler.fit_transform(df[educ_cols])

inertia = []
k_value = range(1, 7)
for k in k_value:
    kmeans = KMeans(random_state=42, n_clusters=k)
    kmeans.fit(scaled_educ)
    inertia.append(kmeans.inertia_)

plt.plot(k_value, inertia)
plt.axis([1, 6, 13, 390])
plt.title("Inertia per Number of Clusters")
plt.tight_layout()
plt.show()

fig = px.line(x=k_value, y=inertia)
fig.update_layout(
    title="Inertia per Cluster",
    xaxis_title="k",
    yaxis_title="Inertia"
)
fig.show()

# Silhouette score for labor force
sil_score = []
k_value = range(2, 7)
for k in k_value:
    kmeans = KMeans(random_state=42, n_clusters=k)
    kmeans.fit(scaled_educ)
    sil_score.append(silhouette_score(scaled_educ, kmeans.labels_))

plt.plot(k_value, sil_score)
plt.axis([2, 6, 0.54, 0.63])
plt.title("Silhouette Score per Number of Clusters")
plt.tight_layout()
plt.show()

fig = px.line(x=k_value, y=sil_score)
fig.update_layout(
    title="Score per Cluster",
    xaxis_title="k",
    yaxis_title="Silhouette Score"
)
fig.show()

# Applying the optimal score for kmeans.
kmeans = KMeans(n_clusters=2, random_state=42)
df["educ_cluster"] = kmeans.fit_predict(df[educ_cols])

# Visualizing secondary education by cluster
sns.scatterplot(data=df, x="Secondary Education(f)",
                y="Secondary Education(m)", hue="educ_cluster")
plt.title("Cluster of Secondary Education")
plt.tight_layout()
plt.show()

# Visualizing labor force by cluster
sns.scatterplot(data=df, x="Labour Force(f)",
                y="Labour Force(m)", hue="lf_cluster")
plt.title("Cluster of Labour Force")
plt.tight_layout()
plt.show()

# Analyzing the result of kmeans for labor force
lf_cluster_1 = df[df["lf_cluster"] == 0]
lf_cluster_2 = df[df["lf_cluster"] == 1]
lf_cluster_3 = df[df["lf_cluster"] == 2]

df["lf_cluster"].value_counts()

# Visualing data from each cluster.
num_cols = list(df.select_dtypes(include=["float", "int"])
                .drop(columns=["HDI Rank", "lf_cluster", "educ_cluster"],
                      axis=1).columns)
len(num_cols)

def distribution_plot(data, columns, title):
    plt.figure(figsize=(15, 12))
    plt.suptitle(f"Feature Distribution for {title}")
    for i, cols in enumerate(columns):
        plt.subplot(3, 3, i+1)
        sns.histplot(data, x=cols)
        plt.title(cols)
        plt.xlabel("")
        plt.ylabel("")
    plt.tight_layout()
    plt.show()

distribution_plot(lf_cluster_1, num_cols, title="Cluster 1")
distribution_plot(lf_cluster_2, num_cols, title="Cluster 2")
distribution_plot(lf_cluster_3, num_cols, title="Cluster 3")

educ_cluster_1 = df[df["educ_cluster"] == 0]
educ_cluster_2 = df[df["educ_cluster"] == 1]

distribution_plot(educ_cluster_1, num_cols, title="Cluster 1")
distribution_plot(educ_cluster_2, num_cols, title="Cluster 2")


# ================================================================
# Visualization with TSNE
# ================================================================
tsne_df = df.copy()

num_cols = list(tsne_df.select_dtypes(include=["float", "int"])
                .drop(columns=["HDI Rank", "GII Rank", "lf_cluster",
                               "educ_cluster"], axis=1).columns)
len(num_cols)

scaler = StandardScaler()
scaled_tsne_df = scaler.fit_transform(tsne_df[num_cols])

# 2D t-SNE
"""The perplexity is a essential parameter as it significantly influences
    the outcome of the dimensionality reduction.
    """
divergence = []
perplexity = np.arange(5, 60, 5)
for plex in perplexity:
    tsne = TSNE(n_components=2, init="pca", perplexity=plex, random_state=42)
    tsne.fit(scaled_tsne_df)
    divergence.append(tsne.kl_divergence_)

fig = px.line(x=perplexity, y=divergence, markers=True)
fig.update_layout(xaxis_title="Perplexity", yaxis_title="Divergence")
fig.update_traces(line_color="blue", line_width=1)
fig.show()

# Applying the optimal perplexity.
tsne = TSNE(n_components=2, perplexity=45, random_state=42)
tsne_2d_value = tsne.fit_transform(scaled_tsne_df)

for comp in range(0, 2):
    tsne_df["tsne_2d_" + str(comp+1)] = tsne_2d_value[:, comp]

# 3D t-SNE
divergence = []
perplexity = np.arange(5, 60, 5)
for plex in perplexity:
    tsne = TSNE(n_components=3, init="pca", perplexity=plex, random_state=42)
    tsne.fit(scaled_tsne_df)
    divergence.append(tsne.kl_divergence_)

fig = px.line(x=perplexity, y=divergence, markers=True)
fig.update_layout(xaxis_title="Perplexity", yaxis_title="Divergence")
fig.update_traces(line_color="blue", line_width=1)
fig.show()

# Applying the optimal perplexity
tsne = TSNE(n_components=3, perplexity=40, random_state=42)
tsne_3d_value = tsne.fit_transform(scaled_tsne_df)

for comp in range(0, 3):
    tsne_df["tsne_3d_" + str(comp+1)] = tsne_3d_value[:, comp]

tsne_df.to_csv("../data/processed/gender_tsne_data.csv", index=False)