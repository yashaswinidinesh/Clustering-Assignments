# Clustering Assignments – Notebooks (a–i)

This repo contains my solutions for the clustering assignment from my Data mining learning course.  
Each notebook focuses on a different clustering method or data type (tabular, time series, text, images, audio).



---

## Notebooks

### a) K-Means from scratch


- Implements K-Means using only NumPy (no `sklearn.cluster.KMeans`).
- Includes k-means++ style initialization option.
- Demo on:
  - synthetic blobs
  - Iris dataset (visualized with PCA)
- Uses metrics like:
  - Silhouette score
  - Davies–Bouldin index
  - Calinski–Harabasz index
  - ARI / NMI when labels are available

---

### b) Hierarchical clustering


- Agglomerative (bottom-up) clustering.
- Dendrograms using SciPy (`linkage`, `dendrogram`).
- Different linkages: ward, complete, average, single.
- Example on Iris and synthetic data.
- Same set of clustering quality metrics as above.

---

### c) Gaussian Mixture Models (GMM)


- Uses `sklearn.mixture.GaussianMixture` for soft clustering.
- Compares different numbers of components using AIC/BIC.
- Works well on elliptical clusters.
- Evaluates GMM results with the usual clustering metrics.

---

### d) DBSCAN with PyCaret


- Uses `pycaret.clustering` to run DBSCAN.
- Example on a non-linear dataset (e.g. two moons).
- Handles noise points (label `-1`) correctly in the metrics.
- Shows how DBSCAN can find arbitrarily shaped clusters.

---

### e) Anomaly detection with PyOD


- Shows anomaly detection for:
  - Univariate time series with injected spikes
  - Multivariate data with injected outliers
- Uses PyOD models like:
  - Isolation Forest
  - ECOD
  - KNN
- Reports precision, recall and F1 when ground truth labels for anomalies are known.

---

### f) Time-series clustering with pretrained models
`

- Converts 1D time series into image-like representations (e.g. Gramian Angular Fields).
- Extracts embeddings using a pretrained vision backbone (e.g. ResNet).
- Clusters those embeddings with K-Means and/or HDBSCAN.
- Visualizes the embeddings with UMAP or PCA.

---

### g) Document clustering with LLM / sentence embeddings


- Uses `sentence-transformers` (LLM-style text embeddings) to embed documents.
- Example on a subset of the 20 Newsgroups dataset.
- Clustering with:
  - K-Means (fixed number of clusters)
  - HDBSCAN (lets the algorithm choose the number of clusters)
- Evaluates with:
  - Silhouette score
  - ARI / NMI against the newsgroup labels
- UMAP visualization for a 2D view of the embeddings.

---

### h) Image clustering with image embeddings


- Computes embeddings for images using a pretrained vision model
  (e.g. ImageBind, CLIP, or similar depending on environment).
- Example dataset: CIFAR-10 or another small labeled image set.
- Clusters embeddings using K-Means / HDBSCAN.
- Evaluates clustering vs. ground-truth classes with ARI / NMI.
- 2D plots of embeddings with UMAP.

---

### i) Audio clustering with audio embeddings

- Uses a small speech/audio dataset (e.g. YESNO).
- Extracts embeddings using an audio model
  (ImageBind audio encoder or an MFCC-based feature pipeline if needed).
- Clusters the audio clips and evaluates how well clusters align with simple labels.
- Visualizes the embedding space with UMAP and colors by cluster.


