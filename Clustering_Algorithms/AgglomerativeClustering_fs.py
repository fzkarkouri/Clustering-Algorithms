"""
Implementation of Agglomerative Clustering [https://en.wikipedia.org/wiki/Single-linkage_clustering#Naive_algorithm]
Input : A pandas dataframe containing numerical data (labels should be transformed to index) or numpy array 
Output: A Linkage matrix a four dimensional array 
        [Cluster i,Cluster j,Distance between Cluster i  and Cluster j, Number of Elements]
        Cluster i and Cluster j can be singeltons.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,squareform

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
        
    
    #Signle linkage 
    def single(self,X):
        '''
        Computes the linkage matrix using a single-linkage criterion: At iteration i the most similar pair of clusters (r,s) is the pair that has the smallest distance between clusters.
        Input : X : A pandas dataframe containing numerical data (labels should be transformed to index) or numpy array.
        Output: A Linkage matrix a four dimensional array 
                [Cluster i,Cluster j,Distance between Cluster i  and Cluster j, Number of Elements]
                Cluster i and Cluster j can be singeltons.
        '''
        
        #Compute the Similarity Matrix
        np.fill_diagonal(self.similarity_matrix, np.inf)
        self.clusters = [[i,1] for i in range(self.similarity_matrix.shape[0])]  #Clusters nbr of elements
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
            self.clusters.append([m,nb_el])
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


        
    
    
    
 