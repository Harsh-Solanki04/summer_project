import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

def load_data(file_path="C:\\Users\\harsh\\OneDrive\\Desktop\\Summer Project-KMeans\\Country_data.csv"):
    data=pd.read_csv(file_path)
    return data

def standardize_data(data):
    features=data.drop(columns=['country'])
   
    scaler=StandardScaler()
    scaled_features=scaler.fit_transform(features)
    scaled_data=pd.DataFrame(scaled_features,columns=features.columns)
    return scaled_data

def plot_distributions(scaled_data):
    plt.figure(figsize=(15,10))
    for i,column in enumerate(scaled_data.columns,1):
        plt.subplot(3,4,i)
        sns.histplot(scaled_data[column],kde=True)
        plt.title(f'{column}Distribution')
    plt.tight_layout()
    plt.show()

def find_optimal_clusters(scaled_data, max_k=10):
    best_k = 2
    best_score = -1
    inertia = []
    silhouette_scores = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(scaled_data)
        inertia.append(kmeans.inertia_)
        
        # Calculate silhouette score
        score = silhouette_score(scaled_data, labels)
        silhouette_scores.append(score)
        
        # Track the best silhouette score
        if score > best_score:
            best_score = score
            best_k = k
        
        print(f"Silhouette Score for k={k}: {score:.2f}")

    # Plot elbow and silhouette scores
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), inertia, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores')
    plt.tight_layout()
    plt.show()
    
    print(f"Optimal number of clusters based on silhouette score: {best_k}")
    return best_k

#implementing k-means++ initialisation manually
def kmeans_pp_init(data,k):
    np.random.seed(42)      #for reproducibility
    n_samples=data.shape[0]
    #1: Randomly Pick the first center
    centres=[data[np.random.choice(n_samples)]]
    #2:select the remaining the k-1 centres
    for _ in range(1,k):
        # compute distances from each point to the nearest center
        distances=np.min(cdist(data,np.array(centres)),axis=1)
        #compute probabilities proportional to the squared distances
        probabilities=distances**2 /np.sum(distances**2)
        #randomly select the next center based on probabilities
        next_center=data[np.random.choice(n_samples,p=probabilities)]
        centres.append(next_center)

    return np.array(centres)

#Now perform k-means clustering using the initialised centres
def kmeans_clustering(data,k,init_centres,max_iters=100,tol=1e-4):
    centres=init_centres
    for _ in range(max_iters):
        #1: assign each point to the nearest center
        distances=cdist(data,centres)
        labels=np.argmin(distances,axis=1)
        #2: update centers based on mean of assigned points
        new_centres=np.array([data[labels==i].mean(axis=0) for i in range(k)])

        #convergence check
        if np.linalg.norm(new_centres-centres)<tol:
            break
        centres=new_centres

    return labels,centres

#visualise clusters in 2D using PCA

def visualise_clusters(data,labels,centres,pca_data):
    plt.figure(figsize=(10,8))
    sns.scatterplot(x=pca_data[:,0],y=pca_data[:,1],hue=labels,palette="Set2",s=50)
    plt.scatter(centres[:,0],centres[:,1],c='red',marker='X',s=200,label='Centers')
    plt.title("Clusters and Centers")
    plt.legend()
    plt.show()




if __name__=="__main__":
    country_data=load_data()
    scaled_data = standardize_data(country_data)
    scaled_data_array=scaled_data.to_numpy() #convert to numpy aaray for calculations
    #Find optimal number of clusters
    optimal_k = find_optimal_clusters(scaled_data, max_k=10)
    #initialise clusters using k-means++
    initial_centers=kmeans_pp_init(scaled_data_array,optimal_k)
    labels,centers=kmeans_clustering(scaled_data_array,optimal_k,initial_centers)
    # Add PCA for 2d visualisation
    pca=PCA(n_components=2)
    pca_data=pca.fit_transform(scaled_data_array)
    pca_centers=pca.transform(centers)

    #visualise clusters
    visualise_clusters(scaled_data,labels,pca_centers,pca_data)

    