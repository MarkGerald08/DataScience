import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

plt.rcParams["figure.dpi"] = 600

# =====================================================
# Data loading
# =====================================================
df = pd.read_csv("../data/processed/gender_tsne_data.csv")
df.head(10)

# Visualizing 2D t-SNE.
sns.scatterplot(data=df, x="tsne_2d_1", y="tsne_2d_2", hue="lf_cluster")
plt.tight_layout()
plt.show()

# Visualizing 3D t-SNE
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df["lf_cluster"].unique():
    subset = df[df["lf_cluster"] == c]
    ax.scatter(subset["tsne_3d_1"], subset["tsne_3d_2"], subset["tsne_3d_3"],
               label=f"Cluster {c}", s=50)
ax.set_title("3D t-SNE Visualization", size=20)
ax.set_xlabel("tsne_3d_1")
ax.set_ylabel("tsne_3d_2")
ax.set_zlabel("tsne_3d_3")
plt.legend()
plt.show()

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df["educ_cluster"].unique():
    subset = df[df["educ_cluster"] == c]
    ax.scatter(subset["tsne_3d_1"], subset["tsne_3d_2"], subset["tsne_3d_3"],
               label=f"Cluster {c}", s=50)
ax.set_title("3D t-SNE Visualization", size=20)
ax.set_xlabel("tsne_3d_1")
ax.set_ylabel("tsne_3d_2")
ax.set_zlabel("tsne_3d_3")
plt.legend()
plt.show()