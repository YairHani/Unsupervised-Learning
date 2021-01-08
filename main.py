from time import time

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from kneed import KneeLocator
from mst_clustering import MSTClustering
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns; sns.set()

mpl.rcParams['agg.path.chunksize'] = 10000

def extract_file(file_name,seperate_symbol=','):
    df = pd.read_csv(file_name,sep=seperate_symbol)
    return df
def e_shop_clothing_2008_cleaning_data(df):
    print(df.keys())
    dictionary_of_page_1 = df["page 2 (clothing model)"].unique()
    dictionary_of_page_1 = {dictionary_of_page_1[index]:index for index in range (len(dictionary_of_page_1))}
    df["page 2 (clothing model)"].replace(dictionary_of_page_1, inplace=True)

    # the result is that there is only 1 year, so we can remove this column
    # print(df["year"].unique())
    # i think we can remove both month and day because they dont give more information which distinct between classes

    df = df.drop(df.columns[[0, 1, 2]], axis=1)  # df.columns is zero-based pd.Index

    # I need to understand this column, however garbage in garbage out, because of that I prefer to remove this column for now
    the_classification = df["country"]
    print(list(set(the_classification)))
    df = df.drop(df.columns[[0,1,2]], axis=1)  # df.columns is zero-based pd.Index
    print(df.keys())
    print(df["page 2 (clothing model)"].unique())    # with pd.option_context('display.max_columns', None):  # more options can be specified also
    df = df.drop(df.columns[[1]], axis=1)  # df.columns is zero-based pd.Index
    return df, the_classification

def split_dataset(df,the_classification,thrashold_train_verify = 0.7,thrashold_tv_testing = 0.7):
    X_train, X_test, y_train, y_test = train_test_split(df, the_classification, train_size=thrashold_tv_testing)
    return X_train, y_train, X_test, y_test
def create_graph(SSE, y_text = "SSE", start_point = 1):

    plt.figure()
    plt.style.use("fivethirtyeight")
    amount_of_classes = [i+start_point for i in range(len(SSE))]
    plt.plot(amount_of_classes, SSE, 'bo',
             amount_of_classes, SSE, 'k')
    plt.xlabel("Number of Clusters")
    plt.ylabel(y_text)
    plt.show()

def KMeans_algo(train_df, train_label):
    SSE = []
    train_df = PCA(2, svd_solver="full").fit_transform(train_df)

    for i in range(1,10):
        KMeans_cluster = KMeans(n_clusters=i)
        KMeans_cluster.fit(train_df)
        SSE.append(KMeans_cluster.inertia_)
        labels = KMeans_cluster.fit_predict(X=train_df)
        x = [item[0] for item in train_df]
        y = [item[1] for item in train_df]
        plt.scatter(x, y, c=labels, cmap=cm.jet)

        plt.title("K-Means"+str(i)+" scatter")
        plt.show()
    kn = KneeLocator([i+1 for i in range(len(SSE))], SSE, curve='convex', direction='decreasing')
    print(kn.elbow)
    create_graph(SSE,y_text="Elbow Method")
def GMM_algo(df,label):
    df = PCA(2, svd_solver="full").fit_transform(df)
    silhouette_score_list = []
    for i in range (2,10):
        GMM_cluster = GaussianMixture(n_components=i, random_state=0)
        GMM_cluster.fit(df)
        print(GMM_cluster.predict(df))
        cluster_of_each_point_in_data = GMM_cluster.predict(df)
        silhouette_score_list.append(silhouette_score(df,cluster_of_each_point_in_data, metric='euclidean'))
        print("Silhouette: ",silhouette_score(df,cluster_of_each_point_in_data, metric='euclidean'))
        x = [item[0] for item in df]
        y = [item[1] for item in df]
        plt.title("GMM - "+str(i)+" scatter")
        plt.scatter(x, y, c=cluster_of_each_point_in_data, cmap=cm.jet)
        plt.show()
    y_text = "Silhouette Score"
    create_graph(silhouette_score_list,y_text,start_point=2)
    # I got the best silhouette score with n_components = 2 with silhouette score = 0.355
def DBSCAN_algo(X_train, y_train):
    silhouette_score_list = []
    X_train = PCA(2, svd_solver="full").fit_transform(X_train)
    for i in range (2,20):
        DBSCAN_cluster = DBSCAN(eps=3, min_samples=i)
        labels = DBSCAN_cluster.fit_predict(X_train,y_train)
        # print(GMM_cluster.means_)
        # print(DBSCAN_cluster.predict(df))
        # SSE.append()
        # cluster_of_each_point_in_data = DBSCAN_cluster.predict(df)
        x = [item[0] for item in X_train]
        y = [item[1] for item in X_train]
        plt.scatter(x, y, c=labels, cmap=cm.jet)
        try:
            if(len(list(set(labels))) > 1):
                silhouette_score_list.append(metrics.silhouette_score(X_train, labels, metric='euclidean'))
            else:
                silhouette_score_list.append(-1)
        except:
            print("silhouette_score did not work")

        # print("Silhouette: ",silhouette_score(df,cluster_of_each_point_in_data))

        #Computing "the Silhouette Score"
        # print("Silhouette Coefficient: %0.3f"
        #   % metrics.silhouette_score(X_train, labels, metric='euclidean'))

        print(labels)
    create_graph(silhouette_score_list, y_text="SSE", start_point=2)
def PRIM_algo(X_train, y_train):
    # predict the labels with the MST algorithm
    silhouette_score_list = []
    X_train = PCA(2, svd_solver="full").fit_transform(X_train)
    for i in range (2,10):
        model = MSTClustering(cutoff_scale=i)
        labels = model.fit_predict(X_train,y_train)
        plt.title(str(i)+" scatter")
        x = [item[0] for item in X_train]
        y = [item[1] for item in X_train]
        print("this is x: ",x)
        print("this is y: ",y)
        plt.scatter(x, y, c=labels, cmap=cm.jet)
        plt.show()
        try:
            if(len(list(set(labels))) > 1):
                silhouette_score_list.append(metrics.silhouette_score(X_train, labels, metric='euclidean'))
            else:
                silhouette_score_list.append(-1)
        except:
            print("silhouette_score did not work")
        # print("Silhouette: ",silhouette_score(df,cluster_of_each_point_in_data))

        # #Computing "the Silhouette Score"
        # print("Silhouette Coefficient: %0.3f"
        #       % metrics.silhouette_score(X_train, labels, metric='euclidean'))
        print(labels)
    if(len(silhouette_score_list)!=0):
        kn = KneeLocator([i+1 for i in range(len(silhouette_score_list))], silhouette_score_list, curve='convex', direction='decreasing')
        print(kn.elbow)
    create_graph(silhouette_score_list, y_text="SSE", start_point=2)
def Spectral_algo(X_train,y_train):
    silhouette_score_list = []
    y_text = "SSE"
    X_train = PCA(2, svd_solver="full").fit_transform(X_train)
    for i in range (2,10):
        spectral_cluter = SpectralClustering(n_clusters=i,
                                        random_state=0).fit(X_train,y_train)
        silhouette_score_list.append(silhouette_score(X_train,spectral_cluter.labels_, metric='euclidean'))
        labels = spectral_cluter.labels_
        x = [item[0] for item in X_train]
        y = [item[1] for item in X_train]
        print("this is x: ",x)
        print("this is y: ",y)
        plt.title(str(i)+" clusters")
        plt.scatter(x, y, c=labels, cmap=cm.jet)
        plt.show()
        print(silhouette_score(X_train,spectral_cluter.labels_, metric='euclidean'))
        print(spectral_cluter)
    create_graph(silhouette_score_list,y_text,start_point=2)

def run_shop_clothing_2008():
    df = extract_file("e-shop clothing 2008.csv",seperate_symbol=';')
    df, the_classification = e_shop_clothing_2008_cleaning_data(df)
    X_train, y_train, X_test, y_test = split_dataset(df,
                                                     the_classification,
                                                     thrashold_train_verify = 0.7,
                                                     thrashold_tv_testing = 0.005)
    KMeans_algo(df, the_classification)#0.0005
    GMM_algo(X_train,y_train)#0.005
    DBSCAN_algo(X_train,y_train)#0.05
    PRIM_algo(X_train,y_train)
    Spectral_algo(X_train,y_train)


def cleaning_diabetic_dataset(df):
    print(df["payer_code"])
    for col in list(df.columns.values):
        df[col] = df[col].apply(lambda x: np.nan if x == '?' else x)
    print(list(df.columns.values))
    # the classifications are gender and race
    list_to_replace = ["gender","race"]
    for col_name in list_to_replace:
        dict_to_replace = { specialty:idx for idx, specialty in enumerate(df[col_name].unique())}
        df[col_name].replace(dict_to_replace, inplace=True)
        print(df[col_name].unique())
    classification_gender,classification_race = df["gender"], df["race"]
    del df["gender"]
    del df["race"]
    print(list(df.columns.values))
    #encounter_id is not something that can help us identify the clusters.
    del df["encounter_id"]
    # patient_nbr data about a burn injury
    print(df["age"])
    df["age"].replace("[0-10)",5,inplace=True)
    df["age"].replace("[10-20)",15,inplace=True)
    df["age"].replace("[20-30)",25,inplace=True)
    df["age"].replace("[30-40)",35,inplace=True)
    df["age"].replace("[40-50)",45,inplace=True)
    df["age"].replace("[50-60)",55,inplace=True)
    df["age"].replace("[60-70)",65,inplace=True)
    df["age"].replace("[70-80)",75,inplace=True)
    df["age"].replace("[80-90)",85,inplace=True)
    df["age"].replace("[90-100)",95,inplace=True)
    print(df["age"])
    print("amount of nan: ",df['weight'].isnull().sum())
    print("amount of not nan values: ",len(df) - df['weight'].isnull().sum())
    #for this amount of values, i think we can remove this coloumn
    del df["weight"]
    for col in list(df.columns.values):
        print("the col: ",col,"with amount of: ",df[col].isnull().sum()," nan")
    # payer_code is an ID according to the internet, so I can remove it. Moreover its about 60% nan 40,256
    del df["payer_code"]
    dict_to_replace = { specialty:idx for idx, specialty in enumerate(df["medical_specialty"].unique())}
    df["medical_specialty"].replace(dict_to_replace, inplace=True)
    df.replace("[50-60)",55,inplace=True)

    # I will remove the diagnosis 1,2,3 because for now it takes time to clean and I am not sure about it's contribution to better results.
    del df["diag_1"]
    del df["diag_2"]
    del df["diag_3"]

    df["max_glu_serum"].replace("None",np.nan,inplace=True)
    df["A1Cresult"].replace("None",np.nan,inplace=True)

    for col in list(df.columns.values):
        print("the col: ",col,"with amount of: ",df[col].isnull().sum()," nan")

    # I will remove the col max_glu_serum A1Cresult because most of it is nan
    del df["max_glu_serum"]
    del df["A1Cresult"]

    for col in list(df.columns.values):
        print("the col: ",col,"with amount of: ",df[col].isnull().sum()," nan")

    list_to_replace = ["metformin","readmitted","repaglinide","nateglinide","chlorpropamide","glimepiride","acetohexamide","glipizide","glyburide","tolbutamide","pioglitazone","rosiglitazone","acarbose","miglitol","troglitazone","tolazamide","examide","citoglipton","insulin","glyburide-metformin","glipizide-metformin","glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone","change","diabetesMed"]
    for col_name in list_to_replace:
        dict_to_replace = { specialty:idx for idx, specialty in enumerate(df[col_name].unique())}
        df[col_name].replace(dict_to_replace, inplace=True)
        print(df[col_name].unique())
    return classification_gender, classification_race, df
def run_diabetic_data():
    df = extract_file("diabetic_data.csv",seperate_symbol=',')
    classification_gender, classification_race, df = cleaning_diabetic_dataset(df)
    X_train, y_train, X_test, y_test = split_dataset(df,
                                                     classification_gender,
                                                     thrashold_train_verify = 0.7,
                                                     thrashold_tv_testing = 0.005)

    KMeans_algo(df, classification_gender)
    GMM_algo(X_train,y_train)
    DBSCAN_algo(X_train,y_train)
    PRIM_algo(X_train,y_train)
    Spectral_algo(X_train,y_train)

    X_train, y_train, X_test, y_test = split_dataset(df,
                                                     classification_race,
                                                     thrashold_train_verify = 0.7,
                                                     thrashold_tv_testing = 0.005)

    KMeans_algo(df, classification_gender)
    GMM_algo(X_train,y_train)
    DBSCAN_algo(X_train,y_train)
    PRIM_algo(X_train,y_train)
    Spectral_algo(X_train,y_train)
def run_online_shoppers_intention():
    df = extract_file("online_shoppers_intention.csv",seperate_symbol=',')
    def run_online_shoppers_intention(df):

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df.describe())

        for col in list(df.columns.values):
            print("the col: ",col,"with amount of nan: ",df[col].isnull().sum())
            print("the col: ",col,"with amount of unique: ",df[col].unique())

        # I also wanted to check for corelations between the cluster and the columns
        # in order to identify which I should pick and which I should remove

        list_to_replace = ["Month","VisitorType","Weekend","Revenue"]
        for col_name in list_to_replace:
            if col_name == "Month":
                dict_to_replace = { specialty:idx+2 for idx, specialty in enumerate(df[col_name].unique())}
            else:
                dict_to_replace = { specialty:idx for idx, specialty in enumerate(df[col_name].unique())}
            df[col_name].replace(dict_to_replace, inplace=True)
            print(df[col_name].unique())
        cluster_revenue, cluster_weekend, cluster_visitortype = df["Revenue"],df["Weekend"],df["VisitorType"]
        print(cluster_revenue)
        return cluster_revenue, cluster_weekend, cluster_visitortype
    cluster_revenue, cluster_weekend, cluster_visitortype = run_online_shoppers_intention(df)
    X_train, y_train, X_test, y_test = split_dataset(df,
                                                     cluster_weekend,
                                                     thrashold_train_verify = 0.7,
                                                     thrashold_tv_testing = 0.01)
    KMeans_algo(df, cluster_weekend)
    GMM_algo(X_train,y_train)
    DBSCAN_algo(X_train,y_train)
    PRIM_algo(X_train,y_train)
    Spectral_algo(X_train,y_train)

    X_train, y_train, X_test, y_test = split_dataset(df,
                                                     cluster_revenue,
                                                     thrashold_train_verify = 0.7,
                                                     thrashold_tv_testing = 0.01)
    KMeans_algo(df, cluster_weekend)
    GMM_algo(X_train,y_train)
    DBSCAN_algo(X_train,y_train)
    PRIM_algo(X_train,y_train)
    Spectral_algo(X_train,y_train)

    X_train, y_train, X_test, y_test = split_dataset(df,
                                                     cluster_visitortype,
                                                     thrashold_train_verify = 0.7,
                                                     thrashold_tv_testing = 0.01)
    KMeans_algo(df, cluster_weekend)
    GMM_algo(X_train,y_train)
    DBSCAN_algo(X_train,y_train)
    PRIM_algo(X_train,y_train)
    Spectral_algo(X_train,y_train)

run_shop_clothing_2008()
run_diabetic_data()
run_online_shoppers_intention()
