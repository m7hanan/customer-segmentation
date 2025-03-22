import pandas as pd 
df = pd.read_csv(r"C:\Users\user\Desktop\ML\ML PRojects\custemer segmentation\archive\Mall_Customers.csv")
print (df.head())
print (df.info())
print(df.isnull().sum())
df.describe()
import pandas as pd 
df = pd.read_csv(r"C:\Users\user\Desktop\ML\ML PRojects\custemer segmentation\archive\Mall_Customers.csv")
import matplotlib.pyplot as plt
import seaborn as sns

# Plot distributions of Age, Income, and Spending Score
df.hist(figsize=(12,6),bins=20, edgecolor="black")
plt.show()
plt.figure(figsize=(10, 7))  
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c='blue', edgecolors='black')  
plt.xlabel('Annual Income (k$)')  
plt.ylabel('Spending Score (1-100)')  
plt.title('Customer Distribution based on Income & Spending Score')  
plt.show()
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 

# Selecting the features for clustering
X= df[['Annual Income (k$)', 'Spending Score (1-100)']]
# Using the Elbow Method to find the optimal number of clusters
wcss=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8,5))
plt.plot(range(1,11),wcss, marker='o' , linestyle='--' , color='b' )
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method to Determine Optimal k')
plt.show()
# Applying K-Means Clustering
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Adding cluster labels to the dataset
df['Cluster'] = y_kmeans
plt.figure(figsize=(10, 6))

# Plotting the clusters
plt.scatter(X.values[y_kmeans == 0, 0], X.values[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X.values[y_kmeans == 1, 0], X.values[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X.values[y_kmeans == 2, 0], X.values[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X.values[y_kmeans == 3, 0], X.values[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X.values[y_kmeans == 4, 0], X.values[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')

# Plotting centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', marker='X', label='Centroids')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation using K-Means')
plt.legend()
plt.show()
# Add cluster labels to the dataset
df['Cluster'] = kmeans.labels_

# Display first few rows
print(df.head())
# Cluster-wise customer details
for i in range(5):  # 5 clusters
    print(f"\nCustomers in Cluster {i}:")
    print(df[df['Cluster'] == i][['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
from sklearn.preprocessing import LabelEncoder

# Convert Gender (Male/Female) into numerical values (0/1)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male = 1, Female = 0

# Select new features for clustering
X_new = df[['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Apply KMeans Clustering
kmeans_new = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans_new.fit_predict(X_new)

# Display first few rows
print(df.head())
# Cluster-wise average statistics with Age & Gender
cluster_summary_new = df.groupby('Cluster')[['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(cluster_summary_new)
