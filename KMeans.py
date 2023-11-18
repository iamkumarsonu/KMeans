import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from random import randint
import numpy as np

path = r'D:\K Means Manual Implementation\driver-data.csv'
data = pd.read_csv(path)
data.columns
plt.scatter(data.mean_dist_day, data.mean_over_speed_perc)
sns.scatterplot(x=[X for X in data.mean_dist_day],
                y=[Y for Y in data.mean_over_speed_perc],
                palette="deep",
                legend=None            
                ) 
plt.xlabel('mean_dist_day')
plt.ylabel('mean_over_speed_perc')
plt.title('mean_dist_day vs mean_over_speed_perc')
plt.show()
'''K Means Basic Starts'''
'''Used two ALgorithm euclidean and manhattan'''
def euclideanDistance(x,y,*method):
    distance = 0
    if method == 'manhattan':
        for i in range(0,len(x)):
            distance += abs(x[i]-y[i])
        return distance
    else:
        for i in range(0,len(x)):
            distance += (x[i]-y[i])**2  
        return distance**0.5
    
def randomint(K,lenofData):
    return [randint(0,lenofData) for i in range(0,K)]


def centroids(data,columns,K):
    tensor = np.array(data[columns])
    centroidIndex = randomint(K,len(tensor))
    centroid = tensor[centroidIndex]
    centroidDict = {}
    for index in range(0,len(centroid)):
        centroidDict[index] = centroid[index]
    return tensor,centroidDict

def clusterAssignment(tensor,centroidDict,columns):
    noOfCluster = len(centroidDict.keys())
    distanceMatrix = []
    for dataIndex in range(0,len(tensor)):
        distanceMatrix.append([euclideanDistance(centroidDict[i],tensor[dataIndex]) for i in range(0,noOfCluster)])   
    cluster_assignment = [val.index(max(val)) for val in distanceMatrix]
    finalDF = pd.concat([pd.DataFrame(tensor),pd.DataFrame(cluster_assignment,columns=['Cluster'])],axis=1)
    finalDF.columns = columns+['Cluster']
    return finalDF,centroidDict

def centroidReassignment(finalDF):
    uniqueCluster = finalDF['Cluster'].unique()
    newCentroids = {}
    for cluster in uniqueCluster:
        filterDF = finalDF.loc[finalDF['Cluster']==cluster]
        newCentroids[cluster] = [filterDF[col].mean() for col in filterDF.columns[:-1]]
    return newCentroids


class Kmeans():
    def __init__(self,data,columns,K,noIteration):
        self.data = data
        self.columns = columns
        self.K = K
        self.noIteration = noIteration
    def ModelFit(self):
        total_centroids = []
        if self.K==0:
            return "Please Assign 2 or More Clusters"
        else:
            for iterval in range(0,self.noIteration):
                if iterval ==0:
                    tensor,centroidDict = centroids(self.data,self.columns,self.K)
                    finalDF,centroidDict = clusterAssignment(tensor,centroidDict,self.columns)
                    centroidDict = centroidReassignment(finalDF)
                    total_centroids.append(centroidDict)
                else:
                    finalDF,centroidDict = clusterAssignment(tensor,centroidDict,self.columns)
                    centroidDict = centroidReassignment(finalDF)
                    total_centroids.append(centroidDict)
                    print(centroidDict)
        self.data['Cluster'] = finalDF['Cluster']
        return centroidDict,self.data,total_centroids
        
'''K Means Basic Ends'''        
currCentroids,outputData,allCentroids = Kmeans(data,['mean_dist_day','mean_over_speed_perc'],3,10).ModelFit()
    































