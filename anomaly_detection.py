import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import KMeans


def f2_score(scores):
    beta = 2
    anomalies = np.argwhere(scores < np.percentile(scores, 25)).flatten()
    y_true_anomalies = np.where(y_test == 1)[0]
    predicted_anomalies = len(anomalies)
    TP = len(set(anomalies.tolist()) & set(y_true_anomalies))
    FN = len(set(y_true_anomalies) - set(anomalies))
    FP = predicted_anomalies - TP
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    return (1+beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


df = pd.read_csv('insurance_claims.csv')
df = df.replace('Y', 1).replace('N', 0).replace('?', 'no_info')
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
df['incident_date'] = pd.to_datetime(df['incident_date'])
df['pol_inc_days'] = df['incident_date'] - df['policy_bind_date']
df['pol_inc_days'] = df['pol_inc_days'].dt.days
y = df['fraud_reported']
df = df.drop(['fraud_reported',
              'policy_bind_date',
              'incident_date',
              'incident_location',
              '_c39'], axis=1)
non_numeric_columns = df.select_dtypes(exclude=['number']).columns.to_list()
df = pd.get_dummies(df, columns=non_numeric_columns)
X = df.values
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def plot_ddr(ddr, labels):
    outliers_indices = labels.index[labels == 1].tolist()
    normal_indices = [i for i in range(len(ddr)) if i not in outliers_indices]
    fraud = ddr[outliers_indices]
    normal = ddr[normal_indices]
    plt.figure(figsize=(6, 4))
    plt.scatter(fraud[:,0], fraud[:,1], s=5, color='#f54029', label='Krāpšanas gadījumi')
    plt.scatter(normal[:,0], normal[:,1], s=5, color='#1f77b4')
    plt.legend()
    plt.show()

kmeans = KMeans(n_clusters=2, random_state=1).fit(X)
clusters = pd.Series(kmeans.labels_)

pca = PCA(n_components=2).fit_transform(X)
plot_ddr(pca, y)
plot_ddr(pca, clusters)

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=20, random_state=42).fit_transform(X)
plot_ddr(lle, y)
plot_ddr(lle, clusters)


neighbors = range(10, 110, 10)
metrics = ['euclidean', 'manhattan', 'canberra', 'hamming']
f2_lof_max = 0.0
f2_lof = {metric: [] for metric in metrics}
for n in neighbors:
    for metric in metrics:
        lof = LocalOutlierFactor(n_neighbors=n,
                                algorithm='brute',
                                contamination=0.25,
                                metric=metric)
        lof.fit(X_train)
        y_pred_lof = lof.fit_predict(X_test)
        lof_scores = lof.negative_outlier_factor_
        f2_lof_score = f2_score(lof_scores)
        f2_lof[metric].append(f2_lof_score)
        if f2_lof_score > f2_lof_max:
            f2_lof_max = f2_lof_score


plt.figure(figsize=(10, 6))
bar_width = 0.2
index = np.arange(len(neighbors))
colors = ['b', 'g', 'r', 'c']
for i, metric in enumerate(metrics):
    plt.bar(index + i * bar_width, f2_lof[metric], bar_width, label=metric, color=colors[i])
plt.xlabel('Number of neighbors')
plt.ylabel('F2_score')
plt.grid(True)
plt.xticks(index + bar_width * (len(metrics) - 1) / 2, neighbors)
plt.legend(loc='lower right')
plt.show()


nu_values = range(10, 100, 10)
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
f2_ocsvm_max = 0.0
f2_ocsvm = {kernel: [] for kernel in kernels}
for nu in nu_values:
    for kernel in kernels:
        ocsvm = OneClassSVM(gamma='auto', 
                            kernel=kernel, 
                            nu=nu/100, 
                            degree=3)
        ocsvm.fit(X_train)
        ocsvm_scores = ocsvm.score_samples(X_test)
        f2_ocsvm_score = f2_score(ocsvm_scores)
        f2_ocsvm[kernel].append(f2_ocsvm_score)
        if f2_ocsvm_score > f2_ocsvm_max:
            f2_ocsvm_max = f2_ocsvm_score


plt.figure(figsize=(10, 6))
bar_width = 0.2
index = np.arange(len(nu_values))
colors = ['b', 'g', 'r', 'c']
for i, kernel in enumerate(kernels):
    plt.bar(index + i * bar_width, f2_ocsvm[kernel], bar_width, label=kernel, color=colors[i])
plt.xlabel('NU * 100')
plt.ylabel('F2_score')
plt.grid(True)
plt.xticks(index + bar_width * (len(kernels) - 1) / 2, nu_values)
plt.legend(loc='lower right')
plt.show()


tf.random.set_seed(42)
np.random.seed(42)

best_ae_f2 = 0.0
best_autoencoder = None

input_dim = X_train.shape[1]
experiments = []
encoded_dims = [2, 4, 8, 179, 245, 326]  
batch_sizes = [8, 16, 32]
epochs = [5, 10, 15, 20]
for encoded_dim in encoded_dims:
    for batch_size in batch_sizes:
        for num_epochs in epochs:
            experiment = {}
            experiment['Dim'] = encoded_dim
            experiment['Batch'] = batch_size
            experiment['Epochs'] = num_epochs
            encoding_dim = encoded_dim
            input_data = tf.keras.Input(shape=(input_dim,))
            encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_data)
            decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
            autoencoder = tf.keras.models.Model(input_data, decoded)
            sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
            autoencoder.compile(optimizer=sgd_optimizer, loss='binary_crossentropy')
            autoencoder.fit(X_train, X_train,
                            epochs=num_epochs, 
                            batch_size=batch_size,
                            shuffle=True,
                            verbose=0,
                            validation_data=(X_test, X_test))
            decoded_test_data = autoencoder.predict(X_test)
            ae_scores = np.mean(np.power(X_test - decoded_test_data, 2), axis=1)
            f2_ae = f2_score(ae_scores)
            experiment['F2 Score'] = f2_ae
            experiments.append(experiment)
            if f2_ae > best_ae_f2:
                best_ae_f2 = f2_ae
                best_autoencoder = autoencoder

experiments_df = pd.DataFrame(experiments)
print(experiments_df.to_string(index=False))
max_f2_score_index = experiments_df['F2 Score'].idxmax()
row_with_max_f2_score = experiments_df.loc[max_f2_score_index]
dim = row_with_max_f2_score['Dim']
batch = row_with_max_f2_score['Batch']
epoch = row_with_max_f2_score['Epochs']
f2_exp = row_with_max_f2_score['F2 Score']
print("Autoencoder best result:")
print(f"Encoded Dim: {dim:.0f}")
print(f"Batch Size: {batch:.0f}")
print(f"Epochs: {epoch:.0f}")
print(f"F2 Score: {f2_exp:.2f}")


results = [f2_lof_max, f2_ocsvm_max, f2_exp]

methods = ['LOF', 'OCSVM', 'Autoencoder']

plt.figure(figsize=(8, 6))
plt.bar(methods, results, color='skyblue')
plt.xlabel('Anomaly Detection Method')
plt.ylabel('Maximum F2 Score')
plt.ylim(0, max(results)*1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for method, result in zip(methods, results):
    plt.text(method, result, f'{result:.2f}', ha='center', va='bottom')
plt.show()