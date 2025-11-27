
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# ======================
# 1. CARGAR DATASET
# ======================
df = pd.read_csv("experiments.csv")

print("\nPrimeras filas del dataset:")
print(df.head())

# ======================
# 2. PREPROCESAMIENTO MEJORADO
# ======================
print("\n" + "="*50)
print("ANÁLISIS INICIAL DE DATOS")
print("="*50)

# Verificar outliers en los hiperparámetros
print("Estadísticas de los hiperparámetros:")
print(df[["learning_rate", "epochs", "batch_size"]].describe())

# Usar escalado robusto (menos sensible a outliers)
scaler = RobustScaler()
X_hyperparams = df[["learning_rate", "epochs", "batch_size"]]
X_scaled = scaler.fit_transform(X_hyperparams)

# ======================
# 3. SELECCIÓN ÓPTIMA DE K CON MÚLTIPLES MÉTODOS
# ======================
def find_optimal_k(X, max_k=10):
    """Encuentra el k óptimo usando múltiples métodos"""
    inertia = []
    silhouette_scores = []
    calinski_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = kmeans.fit_predict(X)
        
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        calinski_scores.append(calinski_harabasz_score(X, labels))
    
    return inertia, silhouette_scores, calinski_scores, k_range

inertia, silhouette_scores, calinski_scores, k_range = find_optimal_k(X_scaled)

# Graficar múltiples métricas
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Método del codo
ax1.plot(k_range, inertia, marker='o')
ax1.set_title("Método del Codo")
ax1.set_xlabel("Número de clusters")
ax1.set_ylabel("Inercia")
ax1.grid(True)

# Método de Silhouette
ax2.plot(k_range, silhouette_scores, marker='o', color='green')
ax2.set_title("Método de Silhouette")
ax2.set_xlabel("Número de clusters")
ax2.set_ylabel("Silhouette Score")
ax2.grid(True)

# Método de Calinski-Harabasz
ax3.plot(k_range, calinski_scores, marker='o', color='red')
ax3.set_title("Método Calinski-Harabasz")
ax3.set_xlabel("Número de clusters")
ax3.set_ylabel("Calinski-Harabasz Score")
ax3.grid(True)

plt.tight_layout()
plt.show()

# Seleccionar k óptimo automáticamente
best_k_silhouette = k_range[np.argmax(silhouette_scores)]
best_k_calinski = k_range[np.argmax(calinski_scores)]

print(f"\n🔍 K óptimo según Silhouette: {best_k_silhouette}")
print(f"🔍 K óptimo según Calinski-Harabasz: {best_k_calinski}")

# Usar el mejor k (puedes ajustar manualmente si es necesario)
best_k = best_k_silhouette  # o el que prefieras
print(f"🎯 Usando K = {best_k}")

# ======================
# 4. CLUSTERING MEJORADO
# ======================
# Opción 1: K-Means con parámetros optimizados
kmeans_optimized = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20,           # Más inicializaciones
    max_iter=500,        # Más iteraciones
    tol=1e-6,           # Tolerancia más estricta
    algorithm='elkan'    # Algoritmo más eficiente
)

df["cluster"] = kmeans_optimized.fit_predict(X_scaled)

# ======================
# 5. FILTRADO DE OUTLIERS INTRA-CLUSTER (CORREGIDO)
# ======================
def filter_cluster_outliers(df, cluster_col='cluster', accuracy_col='por_successes', z_threshold=2):
    """Filtra outliers dentro de cada cluster basado en accuracy - VERSIÓN CORREGIDA"""
    df_clean = df.copy()
    df_clean['is_outlier'] = False  # Inicializar la columna
    
    for cluster_id in df[cluster_col].unique():
        cluster_data = df[df[cluster_col] == cluster_id]
        
        # Calcular z-scores del accuracy dentro del cluster
        z_scores = np.abs(stats.zscore(cluster_data[accuracy_col]))
        
        # Identificar outliers - CORRECCIÓN AQUÍ
        outliers_mask = z_scores > z_threshold
        outlier_indices = cluster_data[outliers_mask].index
        
        # CORRECCIÓN: Usar len() en lugar de .sum() para Index
        print(f"Cluster {cluster_id}: {len(outlier_indices)} outliers detectados")
        
        # Marcar outliers
        df_clean.loc[outlier_indices, 'is_outlier'] = True
    
    return df_clean

# Aplicar filtrado de outliers CORREGIDO
df_clean = filter_cluster_outliers(df)

# ======================
# 6. RE-CLUSTERING SIN OUTLIERS (OPCIONAL)
# ======================
# Si quieres clusters aún más puros, re-clusteriza sin outliers
if df_clean['is_outlier'].sum() > 0:
    print(f"\n🔄 Re-clusterizando sin {df_clean['is_outlier'].sum()} outliers...")
    X_clean = X_scaled[~df_clean['is_outlier']]
    kmeans_clean = KMeans(n_clusters=best_k, random_state=42, n_init=15)
    clean_labels = kmeans_clean.fit_predict(X_clean)
    df_clean.loc[~df_clean['is_outlier'], 'cluster_clean'] = clean_labels
else:
    df_clean['cluster_clean'] = df_clean['cluster']
    print("\n✅ No se encontraron outliers significativos")

# ======================
# 7. ANÁLISIS DE CALIDAD DE CLUSTERS
# ======================
print("\n" + "="*60)
print("EVALUACIÓN DE CALIDAD DE CLUSTERS")
print("="*60)

# Métricas de calidad
silhouette_avg = silhouette_score(X_scaled, df_clean['cluster'])
calinski_avg = calinski_harabasz_score(X_scaled, df_clean['cluster'])

print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Calinski-Harabasz Score: {calinski_avg:.4f}")

# Análisis de cohesión intra-cluster
print("\n📊 COHESIÓN INTRA-CLUSTER:")
for cluster_id in range(best_k):
    cluster_data = df_clean[df_clean['cluster'] == cluster_id]
    cluster_points = X_scaled[df_clean['cluster'] == cluster_id]
    
    # Distancia promedio al centroide
    centroid = kmeans_optimized.cluster_centers_[cluster_id]
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    avg_distance = np.mean(distances)
    
    print(f"Cluster {cluster_id}: {len(cluster_data)} puntos, Distancia avg: {avg_distance:.4f}")

# ======================
# 8. ANÁLISIS DETALLADO POR CLUSTER MEJORADO
# ======================
print("\n" + "="*60)
print("ANÁLISIS DETALLADO POR CLUSTER (MEJORADO)")
print("="*60)

for cluster_id in range(best_k):
    cluster_data = df_clean[df_clean['cluster'] == cluster_id]
    accuracy_mean = cluster_data["por_successes"].mean()
    accuracy_std = cluster_data["por_successes"].std()
    lr_mean = cluster_data["learning_rate"].mean()
    epochs_mean = cluster_data["epochs"].mean()
    batch_mean = cluster_data["batch_size"].mean()
    
    # Calcular rango intercuartílico para identificar variabilidad
    Q1 = cluster_data["por_successes"].quantile(0.25)
    Q3 = cluster_data["por_successes"].quantile(0.75)
    IQR = Q3 - Q1
    
    print(f"\n🎯 CLUSTER {cluster_id} (n={len(cluster_data)})")
    print(f"   Hiperparámetros: LR={lr_mean:.6f} | Epochs={epochs_mean:.1f} | Batch={batch_mean:.1f}")
    print(f"   Accuracy: {accuracy_mean:.3f} ± {accuracy_std:.3f}")
    print(f"   IQR Accuracy: [{Q1:.3f} - {Q3:.3f}]")
    print(f"   Rango total: [{cluster_data['por_successes'].min():.3f} - {cluster_data['por_successes'].max():.3f}]")
    
    # Evaluación de estabilidad
    cv = accuracy_std / accuracy_mean if accuracy_mean > 0 else 0  # Coeficiente de variación
    if cv < 0.1:
        stability = "MUY ESTABLE"
    elif cv < 0.2:
        stability = "ESTABLE" 
    else:
        stability = "VARIABLE"
    
    print(f"   Estabilidad: {stability} (CV: {cv:.3f})")

# ======================
# 9. GRÁFICAS MEJORADAS
# ======================

colors = ['#0000FF',  # Azul
          '#FF0000',  # Rojo
         '#FFFF00',  # Amarillo
          '#00FF00']  # Verde

# ============================
# FIGURA 1 — Clusters con Outliers
# ============================

plt.figure(figsize=(10, 8))
ax1 = plt.axes(projection='3d')

for cluster_id in range(best_k):
    cluster_data = df_clean[df_clean['cluster'] == cluster_id]
    normal_data = cluster_data[~cluster_data['is_outlier']]
    outlier_data = cluster_data[cluster_data['is_outlier']]
    
    # Puntos normales
    ax1.scatter(
        normal_data["learning_rate"],
        normal_data["epochs"], 
        normal_data["batch_size"],
        c=colors[cluster_id],
        label=f'Cluster {cluster_id}',
        s=60,
        alpha=0.7
    )
    
    # Outliers en negro
    if len(outlier_data) > 0:
        ax1.scatter(
            outlier_data["learning_rate"],
            outlier_data["epochs"], 
            outlier_data["batch_size"],
            c='black',
            marker='x',
            s=100,
            alpha=1.0
        )

ax1.set_xlabel("Learning Rate")
ax1.set_ylabel("Epochs")
ax1.set_zlabel("Batch Size")
ax1.set_title("Clusters Hiperparámetros")
ax1.legend()
plt.show()


# ============================
# FIGURA 2 — Hiperparámetros vs Accuracy
# ============================

plt.figure(figsize=(10, 8))
ax2 = plt.axes(projection='3d')

sc = ax2.scatter(
    df_clean["learning_rate"],
    df_clean["epochs"],
    df_clean["batch_size"], 
    c=df_clean["por_successes"],
    cmap='viridis',
    s=80,
    alpha=0.7
)
ax2.set_xlabel("Learning Rate")
ax2.set_ylabel("Epochs")
ax2.set_zlabel("Batch Size")
ax2.set_title("Hiperparámetros vs Accuracy")
plt.colorbar(sc, ax=ax2, label='Accuracy')
plt.show()


# ============================
# FIGURA 3 — Boxplot de Accuracy
# ============================

plt.figure(figsize=(10, 8))
ax3 = plt.axes()

cluster_accuracies = [df_clean[df_clean["cluster"] == i]["por_successes"] for i in range(best_k)]

box_plots = ax3.boxplot(
    cluster_accuracies, 
    labels=[f'Cluster {i}' for i in range(best_k)],
    patch_artist=True,
    showfliers=False
)

for i, box in enumerate(box_plots['boxes']):
    box.set_facecolor(colors[i])
    box.set_alpha(0.7)
    box.set_edgecolor('black')
    box.set_linewidth(1.5)

for median in box_plots['medians']:
    median.set_color('red')
    median.set_linewidth(2)

ax3.set_ylabel('Accuracy')
ax3.set_title('Distribución de Accuracy')
ax3.grid(True, alpha=0.3)

plt.show()

# ======================
# 10. RANGO DE HIPERPARÁMETROS POR CLUSTER
# ======================
print("\n" + "="*60)
print("RANGOS DE HIPERPARÁMETROS POR CLUSTER")
print("="*60)

for cluster_id in range(best_k):
    cluster_data = df_clean[df_clean['cluster'] == cluster_id]

    lr_min, lr_max = cluster_data["learning_rate"].min(), cluster_data["learning_rate"].max()
    ep_min, ep_max = cluster_data["epochs"].min(), cluster_data["epochs"].max()
    bs_min, bs_max = cluster_data["batch_size"].min(), cluster_data["batch_size"].max()

    print(f"\n🔵 CLUSTER {cluster_id} (n={len(cluster_data)})")
    print(f"   • Learning Rate: [{lr_min:.6f}  →  {lr_max:.6f}]")
    print(f"   • Epochs:        [{ep_min:.0f}     →  {ep_max:.0f}]")
    print(f"   • Batch Size:    [{bs_min:.0f}     →  {bs_max:.0f}]")

# ======================
# 9. GRÁFICAS MEJORADAS (USANDO TRAINING_TIME)
# ======================

colors = ['#0000FF',  # Azul
          '#FF0000',  # Rojo
          '#FFFF00',  # Amarillo
          '#00FF00']  # Verde

# ============================
# FIGURA 1 — Clusters con Outliers (Hiperparámetros)
# ============================

plt.figure(figsize=(10, 8))
ax1 = plt.axes(projection='3d')

for cluster_id in range(best_k):
    cluster_data = df_clean[df_clean['cluster'] == cluster_id]
    normal_data = cluster_data[~cluster_data['is_outlier']]
    outlier_data = cluster_data[cluster_data['is_outlier']]
    
    # Puntos normales
    ax1.scatter(
        normal_data["learning_rate"],
        normal_data["epochs"], 
        normal_data["batch_size"],
        c=colors[cluster_id],
        label=f'Cluster {cluster_id}',
        s=60,
        alpha=0.7
    )
    
    # Outliers en negro
    if len(outlier_data) > 0:
        ax1.scatter(
            outlier_data["learning_rate"],
            outlier_data["epochs"], 
            outlier_data["batch_size"],
            c='black',
            marker='x',
            s=100,
            alpha=1.0
        )

ax1.set_xlabel("Learning Rate")
ax1.set_ylabel("Epochs")
ax1.set_zlabel("Batch Size")
ax1.set_title("Clusters de Hiperparámetros (con Outliers)")
ax1.legend()
plt.show()


# ============================
# FIGURA 2 — Hiperparámetros vs TRAINING_TIME
# ============================

plt.figure(figsize=(10, 8))
ax2 = plt.axes(projection='3d')

sc = ax2.scatter(
    df_clean["learning_rate"],
    df_clean["epochs"],
    df_clean["batch_size"], 
    c=df_clean["training_time"],     # <<< CAMBIO AQUÍ
    cmap='viridis',
    s=80,
    alpha=0.7
)

ax2.set_xlabel("Learning Rate")
ax2.set_ylabel("Epochs")
ax2.set_zlabel("Batch Size")
ax2.set_title("Hiperparámetros vs Training Time")

plt.colorbar(sc, ax=ax2, label='Training Time (s)')   # <<< CAMBIO AQUÍ
plt.show()


# ============================
# FIGURA 3 — Boxplot de TRAINING_TIME
# ============================

plt.figure(figsize=(10, 8))
ax3 = plt.axes()

cluster_times = [df_clean[df_clean["cluster"] == i]["training_time"] for i in range(best_k)]

box_plots = ax3.boxplot(
    cluster_times, 
    labels=[f'Cluster {i}' for i in range(best_k)],
    patch_artist=True,
    showfliers=False
)

for i, box in enumerate(box_plots['boxes']):
    box.set_facecolor(colors[i])
    box.set_alpha(0.7)
    box.set_edgecolor('black')
    box.set_linewidth(1.5)

for median in box_plots['medians']:
    median.set_color('red')
    median.set_linewidth(2)

ax3.set_ylabel('Training Time (s)')
ax3.set_title('Distribución de Training Time por Cluster')
ax3.grid(True, alpha=0.3)

plt.show()

best_row = df_clean.loc[df_clean["por_successes"].idxmax()]
print(f"\n🔵 Mejor configuración encontrada:\nAccuracy = {best_row['por_successes']:.3f}  |  LR = {best_row['learning_rate']}  |  Epochs = {best_row['epochs']}  |  Batch = {best_row['batch_size']}")
