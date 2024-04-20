#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs clustering analysis on thermonuclear burst data from Low-Mass X-ray 
Binaries (LMXBs) to explore and categorize burst characteristics such as Burst Duration, 
Peak Flux, Rise Time, and Decay Time. It aggregates data from multiple sources, each 
labeled by the presence of helium, and calculates additional features like the AUC/Peak
Flux ratio. The analysis employs K-means and Gaussian Mixture Models (GMM) to classify
bursts, using methods such as the Elbow Method and Silhouette Scores to determine the 
optimal number of clusters. The script visualizes clustering results, calculates 
cluster-wise statistics, and assesses classification accuracy against known helium labels.
It also estimates the uncertainty in cluster assignments by analyzing the probabilities 
that data points belong to multiple clusters. The output includes visual insights,
statistical summaries, and an evaluation of the clustering accuracy, providing valuable
insights for scientific analysis of burst data.
@author: oskarniemira
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import os

# Paths to your datasets
dataset_paths = [
    r'/Users/oskarniemira/Desktop/Masters Project/lokalne testy/data/csv data/He bursts/data_4U 1636.csv',
    r'/Users/oskarniemira/Desktop/Masters Project/lokalne testy/data/csv data/He bursts/data_4U 1702.csv',
    r'/Users/oskarniemira/Desktop/Masters Project/lokalne testy/data/csv data/He bursts/data_4U 1728.csv',
    r'/Users/oskarniemira/Desktop/Masters Project/lokalne testy/data/csv data/He bursts/data_4U 1735.csv'

]

is_he_dict = {
    'data_4U 1636.csv': 'not He',
    'data_4U 1702.csv': 'He',
    'data_4U 1728.csv': 'He',
    'data_4U 1735.csv': 'He'
}

dataframes = []
for path in dataset_paths:
    df = pd.read_csv(path)
    # Extract the filename from the path and use it to label the data
    filename = os.path.basename(path)
    df['Burst_Type'] = is_he_dict[filename.split('/')[-1]]  # Use 'He' or 'not He' based on the dictionary
    dataframes.append(df)
data = pd.concat(dataframes, ignore_index=True)

data.describe().transpose()
pd.plotting.scatter_matrix(data[['Burst Duration', 'Peak Flux', 'Rise Time', 'Decay Time', 'AUC']], alpha=0.2, figsize=(12, 12))
plt.show()

# Calculate the ratio between AUC and peak flux
data['AUC/Peak Flux'] = data['AUC'] / data['Peak Flux']

X = data[['Burst Duration', 'Peak Flux', 'Rise Time', 'Decay Time', 'AUC', 'AUC/Peak Flux']].values



# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
silhouette_scores = []
range_n_clusters = list(range(2, 7))  # Example range from 2 to 5
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plotting the Elbow Method graph
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(range_n_clusters, inertia, marker='o')
plt.title('Elbow Method For Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(range_n_clusters)
plt.grid(True)
plt.show()

# Plotting Silhouette Scores to assist with cluster number selection
plt.figure(figsize=(8, 4), dpi=300)
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Scores For Optimal number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range_n_clusters)
plt.grid(True)
plt.show()

he_bursts = data[data['Burst_Type'] == 'He']
not_he_bursts = data[data['Burst_Type'] == 'not He']



# Selecting n_clusters based on the Elbow method and silhouette scores observation
# Let's assume n_clusters is selected here based on above plots
n_clusters = 3 
cluster_labels = {0: 'H', 1: 'He', 2: 'H/He'}

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)
data['Cluster'] = kmeans.labels_

# Calculate the mean of 'AUC/Peak Flux' for each cluster
cluster_means = data.groupby('Cluster')['AUC/Peak Flux'].mean()
print("means" , cluster_means)


# Enhanced visualization with physical interpretation context
features = ['Burst Duration', 'AUC/Peak Flux']
colors = ['green', 'blue', 'red']  
markers = ['+', '.', 'x']  

for i, feature in enumerate(features):
    plt.figure(i,figsize=(8, 6), dpi=300)
    for cluster in range(n_clusters):
        cluster_data = data[data['Cluster'] == cluster]
        label = cluster_labels[cluster]
        color = colors[cluster % len(colors)]  # Use modulo to cycle through colors if there are more clusters than colors
        marker = markers[cluster % len(markers)]  # Same for markers
        plt.scatter(cluster_data[feature], cluster_data['AUC/Peak Flux'], label=label, color=color, marker=marker)
        plt.ylabel('AUC/Peak Flux')
        
    plt.title(f'K-means algorithm with 3 clusters')
    plt.xlabel(feature)
    plt.legend()
    plt.show()


# Print the first few rows of the data with cluster assignments
print(data.head())


burst_durations = data['Burst Duration']

# Calculate mean
mean_burst_duration = burst_durations.mean()
print(f"Mean Burst Duration: {mean_burst_duration}")

# Calculate standard deviation
std_burst_duration = burst_durations.std()
print(f"Standard Deviation of Burst Duration: {std_burst_duration}")

timescale = data['AUC/Peak Flux']
mean_timescale = timescale.mean()
print(f"Mean timescale: {mean_timescale}")



cluster_to_he_type = {0: 'not He', 1: 'not He', 2: 'He'}

# Map cluster labels to 'He' or 'not He'
data['Predicted_Type'] = data['Cluster'].map(cluster_to_he_type)

# Calculate how many are correctly identified
correct_predictions = (data['Burst_Type'] == data['Predicted_Type']).sum()
total_predictions = len(data)
accuracy = correct_predictions / total_predictions

print(f"Correct Predictions: {correct_predictions}")
print(f"Total Predictions: {total_predictions}")
print(f"Accuracy: {accuracy:.2f}")

# Create a GMM instance
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)

gmm_cluster_labels = gmm.predict(X_scaled)
data['GMM_Cluster'] = gmm_cluster_labels
probabilities = gmm.predict_proba(X_scaled)

colors = ['green', 'blue', 'red']  # Define a list of colors for each cluster
markers = ['+', '.', 'x']  # Define a list of markers for each cluster
cluster_labels = {0: 'H', 1: 'He', 2: 'H/He'}
cluster_means = data.groupby('GMM_Cluster')['AUC/Peak Flux'].mean()
print("means" , cluster_means)

features = ['Burst Duration', 'AUC/Peak Flux']
for i, feature in enumerate(features):
    plt.figure(i, dpi=300)
    for cluster in range(n_clusters):
        cluster_data = data[data['GMM_Cluster'] == cluster]
        label = cluster_labels[cluster]
        if feature == 'Burst Duration':
            plt.scatter(cluster_data[feature], cluster_data['AUC/Peak Flux'], label=label)  # Plot against 'AUC/Peak Flux'
            plt.ylabel('AUC/Peak Flux')
        
    plt.title(f'GMM Feature Analysis - {feature} vs. Peak Flux')
    plt.xlabel(feature)
    plt.legend()
    plt.show()

for i, feature in enumerate(features):
    plt.figure(i,figsize=(8, 6), dpi=300)
    for cluster in range(n_clusters):
        cluster_data = data[data['GMM_Cluster'] == cluster]
        label = cluster_labels[cluster]
        color = colors[cluster % len(colors)]  # Use modulo to cycle through colors if there are more clusters than colors
        marker = markers[cluster % len(markers)]  # Same for markers
        plt.scatter(cluster_data[feature], cluster_data['AUC/Peak Flux'], label=label, color=color, marker=marker)
        plt.ylabel('AUC/Peak Flux')
        
    plt.title('GMM algorithm with 3 clusters')
    plt.xlabel(feature)
    plt.legend()
    plt.show()

    
gmm_silhouette_score = silhouette_score(X_scaled, gmm_cluster_labels)
print(f"GMM Silhouette Score: {gmm_silhouette_score}")

gmm_cluster_to_he_type = {0: 'not He', 1: 'He', 2: 'not He'}

# Map GMM cluster labels to 'He' or 'not He'
data['GMM_Predicted_Type'] = data['GMM_Cluster'].map(gmm_cluster_to_he_type)

# Calculate how many bursts were correctly identified by GMM
gmm_correct_predictions = (data['Burst_Type'] == data['GMM_Predicted_Type']).sum()
gmm_total_predictions = len(data)
gmm_accuracy = gmm_correct_predictions / gmm_total_predictions

print(f"GMM Correct Predictions: {gmm_correct_predictions}")
print(f"GMM Total Predictions: {gmm_total_predictions}")
print(f"GMM Accuracy: {gmm_accuracy:.2f}")



print(probabilities[:5])  # Print probabilities for the first 5 points

diff_01 = np.abs(probabilities[:, 0] - probabilities[:, 1])
diff_02 = np.abs(probabilities[:, 0] - probabilities[:, 2])
diff_12 = np.abs(probabilities[:, 1] - probabilities[:, 2])

similar_01 = diff_01 <= 0.05
similar_02 = diff_02 <= 0.05
similar_12 = diff_12 <= 0.05

# Combine the similar flags into a single array
similar_any = similar_01 | similar_02 | similar_12

# Count how many data points have similar probabilities for any pair of clusters
similar_count = np.sum(similar_any)


print("Number of data points with similar probabilities for any cluster pair: ", similar_count)


