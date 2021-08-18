"""
Implementation of Agglomerative Clustering 
[https://en.wikipedia.org/wiki/Single-linkage_clustering#Naive_algorithm]
[https://en.wikipedia.org/wiki/Complete-linkage_clustering#Naive_scheme]
[https://en.wikipedia.org/wiki/UPGMA#Algorithm]
[https://en.wikipedia.org/wiki/Ward%27s_method]

Input : A pandas dataframe containing numerical data (labels should be transformed to index) or numpy array 
Output: A Linkage matrix a four dimensional array 
        [Cluster i,Cluster j,Distance between Cluster i  and Cluster j, Number of Elements]
        Cluster i and Cluster j can be singeltons.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt 
from sklearn import metrics
class AgglomerativeClustering_fs:
    
    #constructor
    def __init__(self, linkage='single'):
        self.similarity_matrix = []
        self.linkage = linkage
        self.L = []
        self.clusters= []
        
    #train model
    def fit(self,X):
        '''
        Computes the similarity_matrix of the observations
        Input : X : A pandas dataframe containing numerical data (labels should be transformed to index) or numpy array.  
        Output: Similarity matrix a N*N matrix where N is the number of observations in X 
                and the element in the row i column j is the distance between the observation i and j.
        '''
        self.similarity_matrix = squareform(pdist(X))
        
        if self.linkage == 'single':
            self.single(X)
        elif self.linkage == 'complete':
            self.complete(X)
        elif self.linkage == 'UPGMA':
            self.UPGMA(X)
        elif self.linkage == 'WPGMA':
            self.WPGMA(X)
        elif self.linkage == 'centroid':
            self.centroid(X)
        elif self.linkage == 'ward':
            self.ward(X)
    
    #Signle linkage 
    def single(self,X):
        '''
        Computes the linkage matrix using a single-linkage criterion: At iteration i the most similar pair of clusters (r,s) is the pair that has the smallest distance between clusters, the distance is the distance between the closest points.
        Input : X : A pandas dataframe containing numerical data (labels should be transformed to index) or numpy array.
        Output: A Linkage matrix a four dimensional array 
                [Cluster i,Cluster j,Distance between Cluster i  and Cluster j, Number of Elements]
                Cluster i and Cluster j can be singeltons.
        '''
        
        #Compute the Similarity Matrix
        np.fill_diagonal(self.similarity_matrix, np.inf)
        self.clusters = [[i,1] for i in range(self.similarity_matrix.shape[0])]  #Clusters nbr of elements
        self.clusters = np.array(self.clusters)
        m = self.similarity_matrix.shape[0]
        nb_el=1
        similarity_matrix_copy = self.similarity_matrix
        
        while(nb_el<self.similarity_matrix.shape[0]):

            rs = np.unravel_index(similarity_matrix_copy.argmin(),similarity_matrix_copy.shape)
            d_rs = similarity_matrix_copy[rs]
            nb_el= self.clusters[rs[0]][1]+self.clusters[rs[1]][1]
            cl = list(rs)
            cl.append(d_rs)
            cl.append(nb_el)
            self.clusters = np.append(self.clusters,[[m,nb_el]],axis=0 )
            m=m+1

            self.L.append(cl)

            new_cluster_values = similarity_matrix_copy[[rs[0],rs[1]]].min(axis=0)
            similarity_matrix_copy = np.concatenate((similarity_matrix_copy,[new_cluster_values]),axis=0)
            new_cluster_values= np.append(new_cluster_values,[np.inf])
            similarity_matrix_copy = np.concatenate((similarity_matrix_copy,np.array([new_cluster_values]).T),axis=1)

            similarity_matrix_copy[:,rs[0]]=np.inf
            similarity_matrix_copy[:,rs[1]]=np.inf
            similarity_matrix_copy[rs[0],:]=np.inf
            similarity_matrix_copy[rs[1],:]=np.inf 
            
        self.L = np.array(self.L)
        
        
    #Complete linkage 
    def complete(self,X):
        '''
        Computes the linkage matrix using a complete-linkage criterion: At iteration i the most similar pair of clusters (r,s) is the pair that has the smallest distance between clusters. The distance is the distance between the furthest points.
        Input : X : A pandas dataframe containing numerical data (labels should be transformed to index) or numpy array.
        Output: A Linkage matrix a four dimensional array 
                [Cluster i,Cluster j,Distance between Cluster i  and Cluster j, Number of Elements]
                Cluster i and Cluster j can be singeltons.
        '''
        
        #Compute the Similarity Matrix
        np.fill_diagonal(self.similarity_matrix, np.inf)
        self.clusters = [[i,1] for i in range(self.similarity_matrix.shape[0])]  #Clusters nbr of elements
        self.clusters = np.array(self.clusters)
        m = self.similarity_matrix.shape[0]
        nb_el=1
        similarity_matrix_copy = self.similarity_matrix
        
        while(nb_el<self.similarity_matrix.shape[0]):

            rs = np.unravel_index(similarity_matrix_copy.argmin(),similarity_matrix_copy.shape)
            d_rs = similarity_matrix_copy[rs]
            nb_el= self.clusters[rs[0]][1]+self.clusters[rs[1]][1]
            cl = list(rs)
            cl.append(d_rs)
            cl.append(nb_el)
            self.clusters = np.append(self.clusters,[[m,nb_el]],axis=0 )
            m=m+1

            self.L.append(cl)

            new_cluster_values = similarity_matrix_copy[[rs[0],rs[1]]].max(axis=0)
            similarity_matrix_copy = np.concatenate((similarity_matrix_copy,[new_cluster_values]),axis=0)
            new_cluster_values= np.append(new_cluster_values,[np.inf])
            similarity_matrix_copy = np.concatenate((similarity_matrix_copy,np.array([new_cluster_values]).T),axis=1)

            similarity_matrix_copy[:,rs[0]]=np.inf
            similarity_matrix_copy[:,rs[1]]=np.inf
            similarity_matrix_copy[rs[0],:]=np.inf
            similarity_matrix_copy[rs[1],:]=np.inf 
            
        self.L = np.array(self.L)
        
     #UPGMA linkage 
    def UPGMA(self,X):
        '''
        Computes the linkage matrix using a UPGMA-linkage criterion: At iteration i the most similar pair of clusters (r,s) is the pair that has the smallest distance between clusters. The distance is the proportional average distance between the point and the two clusters.
        Input : X : A pandas dataframe containing numerical data (labels should be transformed to index) or numpy array.
        Output: A Linkage matrix a four dimensional array 
                [Cluster i,Cluster j,Distance between Cluster i  and Cluster j, Number of Elements]
                Cluster i and Cluster j can be singeltons.
        '''
        
        #Compute the Similarity Matrix
        np.fill_diagonal(self.similarity_matrix, np.inf)
        self.clusters = [[i,1] for i in range(self.similarity_matrix.shape[0])]  #Clusters nbr of elements
        self.clusters = np.array(self.clusters)
        m = self.similarity_matrix.shape[0]
        nb_el=1
        similarity_matrix_copy = self.similarity_matrix
        
        while(nb_el<self.similarity_matrix.shape[0]):

            rs = np.unravel_index(similarity_matrix_copy.argmin(),similarity_matrix_copy.shape)
            d_rs = similarity_matrix_copy[rs]
            nb_el= self.clusters[rs[0]][1]+self.clusters[rs[1]][1]
            cl = list(rs)
            cl.append(d_rs)
            cl.append(nb_el)
            self.clusters = np.append(self.clusters,[[m,nb_el]],axis=0 )
            m=m+1

            self.L.append(cl)
            
            
            
            x0 = np.copy(similarity_matrix_copy[rs[0]])
            x0[x0 == np.inf] = 0
            x1 = np.copy(similarity_matrix_copy[rs[1]])
            x1[x1 == np.inf] = 0

            new_cluster_values = (self.clusters[rs[0]][1] * x0 + self.clusters[rs[1]][1] * x1)/(self.clusters[rs[0]][1]+ self.clusters[rs[1]][1])
            similarity_matrix_copy = np.concatenate((similarity_matrix_copy,[new_cluster_values]),axis=0)
            new_cluster_values= np.append(new_cluster_values,[np.inf])
            similarity_matrix_copy = np.concatenate((similarity_matrix_copy,np.array([new_cluster_values]).T),axis=1)

            similarity_matrix_copy[:,rs[0]]=np.inf
            similarity_matrix_copy[:,rs[1]]=np.inf
            similarity_matrix_copy[rs[0],:]=np.inf
            similarity_matrix_copy[rs[1],:]=np.inf 
            similarity_matrix_copy[similarity_matrix_copy == 0]=np.inf
        self.L = np.array(self.L)
        
        
    #WPGMA linkage 
    def WPGMA(self,X):
        '''
        Computes the linkage matrix using a WPGMA-linkage criterion: At iteration i the most similar pair of clusters (r,s) is the pair that has the smallest distance between clusters. The distance is the average distance between the point and the two clusters.
        Input : X : A pandas dataframe containing numerical data (labels should be transformed to index) or numpy array.
        Output: A Linkage matrix a four dimensional array 
                [Cluster i,Cluster j,Distance between Cluster i  and Cluster j, Number of Elements]
                Cluster i and Cluster j can be singeltons.
        '''
        
        #Compute the Similarity Matrix
        np.fill_diagonal(self.similarity_matrix, np.inf)
        self.clusters = [[i,1] for i in range(self.similarity_matrix.shape[0])]  #Clusters nbr of elements
        self.clusters = np.array(self.clusters)
        m = self.similarity_matrix.shape[0]
        nb_el=1
        similarity_matrix_copy = self.similarity_matrix
        
        while(nb_el<self.similarity_matrix.shape[0]):

            rs = np.unravel_index(similarity_matrix_copy.argmin(),similarity_matrix_copy.shape)
            d_rs = similarity_matrix_copy[rs]
            nb_el= self.clusters[rs[0]][1]+self.clusters[rs[1]][1]
            cl = list(rs)
            cl.append(d_rs)
            cl.append(nb_el)
            self.clusters = np.append(self.clusters,[[m,nb_el]],axis=0 )
            m=m+1

            self.L.append(cl)
            
            
            
            x0 = np.copy(similarity_matrix_copy[rs[0]])
            x0[x0 == np.inf] = 0
            x1 = np.copy(similarity_matrix_copy[rs[1]])
            x1[x1 == np.inf] = 0

            new_cluster_values = (x0 + x1)/2
            similarity_matrix_copy = np.concatenate((similarity_matrix_copy,[new_cluster_values]),axis=0)
            new_cluster_values= np.append(new_cluster_values,[np.inf])
            similarity_matrix_copy = np.concatenate((similarity_matrix_copy,np.array([new_cluster_values]).T),axis=1)

            similarity_matrix_copy[:,rs[0]]=np.inf
            similarity_matrix_copy[:,rs[1]]=np.inf
            similarity_matrix_copy[rs[0],:]=np.inf
            similarity_matrix_copy[rs[1],:]=np.inf 
            similarity_matrix_copy[similarity_matrix_copy == 0]=np.inf
        self.L = np.array(self.L)
        
         
    #Centroid linkage 
    def centroid(self,X):
        '''
        Computes the linkage matrix using a centroid-linkage criterion: At iteration i the most similar pair of clusters (r,s) is the pair that has the smallest distance between clusters. The new distance is the distance between the centroid of the new formed cluster and the centroids of the existing clusters.
        Input : X : A pandas dataframe containing numerical data (labels should be transformed to index) or numpy array.
        Output: A Linkage matrix a four dimensional array 
                [Cluster i,Cluster j,Distance between Cluster i  and Cluster j, Number of Elements]
                Cluster i and Cluster j can be singeltons.
        '''
        
        #Compute the Similarity Matrix
        np.fill_diagonal(self.similarity_matrix, np.inf)
        self.clusters = [[i,1] for i in range(self.similarity_matrix.shape[0])]  #Clusters nbr of elements
        self.clusters = np.array(self.clusters)
        m = self.similarity_matrix.shape[0]
        nb_el=1
        similarity_matrix_copy = np.copy(self.similarity_matrix)
        X_copy = np.copy(X)
        while(nb_el<self.similarity_matrix.shape[0]):

            rs = np.unravel_index(similarity_matrix_copy.argmin(),similarity_matrix_copy.shape)
            d_rs = similarity_matrix_copy[rs]
            nb_el= self.clusters[rs[0]][1]+self.clusters[rs[1]][1]
            cl = list(rs)
            cl.append(d_rs)
            cl.append(nb_el)
            self.clusters = np.append(self.clusters,[[m,nb_el]],axis=0 )
            m=m+1

            self.L.append(cl)
            
            
            
            x0 = np.copy(similarity_matrix_copy[rs[0]])
            x0[x0 == np.inf] = 0
            x1 = np.copy(similarity_matrix_copy[rs[1]])
            x1[x1 == np.inf] = 0
            
            c = (self.clusters[rs[0]][1]*X_copy[rs[0]]+self.clusters[rs[1]][1]*X_copy[rs[1]])/nb_el
            X_copy[rs[0]] = np.inf
            X_copy[rs[1]] = np.inf
                
            new_cluster_values = [np.linalg.norm(c-r) for r in X_copy]
            similarity_matrix_copy = np.concatenate((similarity_matrix_copy,[new_cluster_values]),axis=0)
            new_cluster_values = np.append(new_cluster_values,[np.inf])
            similarity_matrix_copy = np.concatenate((similarity_matrix_copy,np.array([new_cluster_values]).T),axis=1)
            
            
            X_copy = np.append(X_copy,[c], axis=0)

            similarity_matrix_copy[:,rs[0]]=np.inf
            similarity_matrix_copy[:,rs[1]]=np.inf
            similarity_matrix_copy[rs[0],:]=np.inf
            similarity_matrix_copy[rs[1],:]=np.inf 
            similarity_matrix_copy[similarity_matrix_copy == 0]=np.inf
        self.L = np.array(self.L)
        
        
        
    #Centroid linkage 
    def ward(self,X):
        '''
        Computes the linkage matrix using a ward-linkage criterion: At iteration i the most similar pair of clusters (r,s) is the pair that minimizes the within-cluster variance after merge.
        Input : X : A pandas dataframe containing numerical data (labels should be transformed to index) or numpy array.
        Output: A Linkage matrix a four dimensional array 
                [Cluster i,Cluster j,Distance between Cluster i  and Cluster j, Number of Elements]
                Cluster i and Cluster j can be singeltons.
        '''
        
        #Compute the Similarity Matrix
        np.fill_diagonal(self.similarity_matrix, np.inf)
        self.clusters = [[i,1] for i in range(self.similarity_matrix.shape[0])]  #Clusters nbr of elements
        self.clusters = np.array(self.clusters)
        m = self.similarity_matrix.shape[0]
        nb_el=1
        similarity_matrix_copy = np.copy(self.similarity_matrix)
        while(nb_el<self.similarity_matrix.shape[0]):

            rs = np.unravel_index(similarity_matrix_copy.argmin(),similarity_matrix_copy.shape)
            d_rs = similarity_matrix_copy[rs]

            nb_el= self.clusters[rs[0]][1]+self.clusters[rs[1]][1]
            
            # Ward parameters preparation

            nb_els = self.clusters[:,1] + nb_el 
            nb_els_0 = self.clusters[:,1] + self.clusters[rs[0]][1]
            nb_els_1 = self.clusters[:,1] + self.clusters[rs[1]][1]
            nb_els_2 = self.clusters[:,1] * (d_rs**2)

            # Ward parameters preparation
            
            cl = list(rs)
            cl.append(d_rs)
            cl.append(nb_el)
            self.clusters = np.append (self.clusters,[[m,nb_el]],axis=0)
            m=m+1

            self.L.append(cl)
            
            
            
            x0 = np.copy(similarity_matrix_copy[rs[0]])

            x1 = np.copy(similarity_matrix_copy[rs[1]])
            

                
            new_cluster_values = np.sqrt(np.divide((nb_els_0 * np.square(x0)  +  nb_els_1 * np.square(x1) - nb_els_2), nb_els))
            similarity_matrix_copy = np.concatenate((similarity_matrix_copy,[new_cluster_values]),axis=0)
            new_cluster_values = np.append(new_cluster_values,[np.inf])
            similarity_matrix_copy = np.concatenate((similarity_matrix_copy,np.array([new_cluster_values]).T),axis=1)
            
            

            similarity_matrix_copy[:,rs[0]]=np.inf
            similarity_matrix_copy[:,rs[1]]=np.inf
            similarity_matrix_copy[rs[0],:]=np.inf
            similarity_matrix_copy[rs[1],:]=np.inf 
            similarity_matrix_copy[similarity_matrix_copy == 0]=np.inf
        self.L = np.array(self.L)
        
    

    # get the labels of each cluster
    def get_labels(self,max_dist):
        '''
        Get the cluster label of each observation
        Input : max_dist : the maximum distance between two clusters to be merged as one.
        Output: An array of size N where N is the number of original observations in the input dataset X, 
                the element at index i is the cluster label of the observation at index i in the dataset X.
        '''
        n = self.L.shape[0]
        labels = np.zeros((2*n+1,),dtype=int)
        ind = 0 
        
        #determin clusters
        clusters = self.L[self.L[:,2]<=max_dist, :][:,0:2]
        #determin singeltons 
        singltons = self.L[self.L[:,2]>max_dist, :][:,0:2].flatten()
        if singltons.size == 0:
            labels = labels+1
            return labels[:n+1]
        
        for i in range(0,clusters.shape[0]):
            max_i = int(max(clusters[i,0],clusters[i,1]))
            min_i = int(min(clusters[i,0],clusters[i,1]))
            
            #case: original observations
            if(max_i<=n):
                
                
                ind = max(max(labels),ind) + 1 
                labels[max_i]=ind
                labels[min_i]=ind
                labels[n+1+i]=ind
                
            #case: 1 original observation one new cluster    
            elif(min_i<=n):
                
                
                ind = labels[max_i]
                labels[min_i] = ind  
                labels[max_i] = ind
                labels[n+1+i]=ind

            #case: 2 new formed clusters    
            else:
                
                
                ind = min(labels[min_i],labels[max_i])
                max_ind = max(labels[min_i],labels[max_i])
                labels[np.where(labels==max_ind)] = ind
                for j in range(0,labels.shape[0]):
                    labels[j] = labels[j]-1 if labels[j]>max_ind else labels[j]
                
                labels[n+1+i]=ind
                
                        

        singltons = singltons[singltons<n+1]
       # print(singltons)
        ind = max(labels)
        for i in singltons:
            ind = ind + 1
            labels[int(i)] = ind
        
            
        return labels[:n+1]
    
def main():
    
    #Read data 
    country_data = pd.read_csv('../Data/Country-data.csv')
    country_data = country_data.set_index('country')
    
    #Standarization
    sc = StandardScaler()
    X = sc.fit_transform(country_data)
    
    
    for linkage_method in ['single','complete','UPGMA','WPGMA','centroid','ward']:
    
        #Clustering
        model = AgglomerativeClustering_fs(linkage=linkage_method)
        model.fit(X)
        L = model.L

        #Choosing the max distance that optimise the davies bouldin score
        max_distances = list(range(1,int(max(L[:,2]))+1)) #
        davies_bouldin_scores = []

        for dist in max_distances:
            labels = model.get_labels(max_dist=dist)
            davies_bouldin_scores.append(metrics.davies_bouldin_score(X, labels))
            i = davies_bouldin_scores.index(min(davies_bouldin_scores))
            
        print('The linkage method {} minimum score is {} with a maxdistance {}'.format(linkage_method,min(davies_bouldin_scores),max_distances[i]))

        plt.plot(max_distances,davies_bouldin_scores)
        plt.title('Davies Bouldin Scores per Max Distance for the Linkage method {}'.format(linkage_method))
        plt.show()
        
        
        
    
    

if __name__ == "__main__":
    main()

    
        
    
    
    
 