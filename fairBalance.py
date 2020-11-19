import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score,silhouette_samples
from imblearn.over_sampling import SMOTE,SMOTENC

class fairBalance:
    """
    Args:
        data: original train set w/ features, target varialbles, and sensitive attributes
        features: list of features used in model training
        continous_features: list of names of continous features
        drop_featrues: list of features that need to be removed for traing (including the senstive attributes)
        sensitive_attribute: single protected attribute with which to do repair
        target: name of the target variable
        cluster_algo: clustering algorithm used for balancing
            - 'kmeans': KMeans clustering
            - 'agg': Agglomerative Clustering
            - 'spe': Spectral Clustering
            - or provide the model of pre-defined clustering algorithm
        ratio: ratio of lowest silhouette score samples to be filtered out
        knn: fairness budget, number of nearest neighbors used in synthetic samples generation

    """
    def __init__ (self, data, features, continous_features, drop_features, sensitive_attribute, target, cluster_algo = 'kmeans', ratio = 0.25, knn = 5):
        self.data = data.copy()
        self.continous_features = continous_features
        self.sensitive_attribute = sensitive_attribute
        self.features = features
        self.drop_features = drop_features
        self.target = target
        self.cluster_algo = cluster_algo
        self.ratio = ratio
        self.knn = knn

    def fit(self):
        self.cluster()
        self.filter()
        

    def cluster(self):
        # normalize continous features
        scaler = MinMaxScaler()
        X = self.data.drop([self.sensitive_attribute, self.target], axis = 1)
        X[self.continous_features] = scaler.fit_transform(X[self.continous_features])

        ## choose clustering algorithms
        if self.cluster_algo == 'kmeans':
            model = KMeans()
        elif self.cluster_algo == 'agg':
            model = AgglomerativeClustering()
        elif self.cluster_algo == 'spe':
            model = SpectralClustering()
        else:
            ## customized model
            model = self.cluster_algo

        ## choose the optimized number of clusters 
        max_s=-np.inf
        for k in range(2,10):
            model = model.set_params(n_clusters = k)
            model.fit(X)
            groups=model.labels_
            s_score=silhouette_score(X,groups)
            score_list=silhouette_samples(X,groups)
            if(s_score>max_s):
                best_k=k
                max_s=s_score
                best_clist=groups
                best_slist=score_list
        print("Cluster the original dataset into %d clusters:"%best_k)
        self.data['score'] = best_slist
        self.data['group'] = best_clist
        
    def filter(self):
        ## filter out the samples with less silhouette scores
        scores = self.data['score'].tolist()
        s_rank=np.sort(scores)
        idx=int(len(s_rank)*self.ratio)
        threshold = s_rank[idx]
        print("Removing %d samples from the original dataset..."%idx)
        self.X_clean = self.data[self.data['score'] > threshold]

    def new_smote(self, dfi):
        label = self.target
        categorical_features = list(set(dfi.keys().tolist()) - set(self.continous_features))
        categorical_loc = [dfi.columns.get_loc(c) for c in categorical_features if c in dfi.keys()]
       
        ## count the class distribution 
        min_y=dfi[label].value_counts().idxmin()
        max_y=dfi[label].value_counts().idxmax()
        min_X=dfi[dfi[label]==min_y]
        max_X=dfi[dfi[label]==max_y]
        ratio=len(max_X)-len(min_X)
        
        ##get the knn for minority class
        nbrs = NearestNeighbors(n_neighbors=self.knn, algorithm='auto').fit(dfi[self.features])
        dfs=[]
        for j in range(len(min_X)):
            dfj=min_X.iloc[j]
            nn_list=nbrs.kneighbors(np.array(dfj[self.features]).reshape(1,-1),return_distance=False)[0]
            df_nn=dfi.iloc[nn_list]
            dfs.append(df_nn)
        df_nns=pd.concat([dfj for dfj in dfs],ignore_index=True).drop_duplicates()   

        """
        This is a hacking method to directly use the pre-implemented SMOTE algorithm. 

        """
        X_temp=pd.concat([df_nns, min_X], ignore_index=True)
        y_temp=list(np.repeat(1,len(df_nns)))+list(np.repeat(0,len(min_X)))
        
        min_k=max(1,min(self.knn, len(df_nns)-1))
        sm=SMOTENC(categorical_features=categorical_loc,random_state=42, sampling_strategy={1: len(df_nns)+ratio, 0:len(min_X)}, k_neighbors=min_k)
        Xi_res, yi_res = sm.fit_resample(X_temp, y_temp)
        df_res=pd.DataFrame(Xi_res,columns=dfi.keys().tolist())
        df_add=df_res.iloc[len(X_temp):]
        df_add[label]=min_y
        df_new=pd.concat([dfi,df_add],ignore_index=True)
        
        return df_new


    def generater(self):
        dfs=[]
        groups = list(self.X_clean['group'].unique())
        ## generate new samples for each group
        for i in groups:
            dfi=self.X_clean[self.X_clean['group']==i].drop(['group', 'score'],axis=1)
            if(len(dfi[self.target].unique())==1 or len(dfi)==0):
                continue
          
            Xi_res=self.new_smote(dfi)
            dfs.append(Xi_res)

        X_cres=pd.concat([dfi for dfi in dfs],ignore_index=True)

        return X_cres[self.features], X_cres[self.target]