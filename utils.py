import numpy as np
import pandas as pd
#from sklearn.cluster import Birch
from sklearn.cluster import KMeans
#from sklearn import preprocessing
import pickle as pkl

from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

import scipy.cluster.vq
import matplotlib.pyplot as plt
import seaborn as sns

def floor_optim(a):
    down_list = ['подвал', 'цоколь', 'Подвал', 'Цоколь', '1 (Цокольный этаж)', 'фактически на уровне 1 этажа', \
                 '-1', '-2', '0', 'антресоль', 'мезонин']
    add = 0
    res = 2
    if type(a) == str:
        if a.isdigit() == False:
            if a in down_list:
                return 3
            for k in down_list:
                if k in str(a) == True:
                    add = 1
            if ' ' in a:
                a = a.replace(' ', '')
            if 'этаж' in a:
                a = a.replace('этаж', '')
            if '-й' in a:
                a = a.replace('-й', '')
            if '.0' in a:
                a = a.replace('.0', '')

        if a.isdigit() == True:
            a = float(a)
        else:
            a = res

    if type(a) == float:
        if str(a)[-1] == '0':
            a = int(a)

    if type(a) == int:
        if a == 1:
            return 1
        elif a in range(-100, 1, 1):
            return res + add
        elif a in range(1, 1000, 1):
            return res + add
    else:
        return res


# функция подготовки к модели
def prep(df=df):
    df['floor'] = df['floor'].fillna(2)
    df['floor'] = df['floor'].apply(lambda x: floor_optim(x))
    # Определим колонки с пропусками в список
    to_fill_list = df.columns[df.isna().any()].tolist()

    df['need_to_feel'] = df.isna().sum(axis=1) > 0
    tdf = df[['lat', 'lng', 'need_to_feel']]
    tdft = tdf[tdf['need_to_feel'] == True]
    tdff = tdf[tdf['need_to_feel'] == False]
    df['nearest'] = np.nan

    for i in tdft.index.tolist():
        dif = tdff[['lat', 'lng']] - tdft.loc[i][['lat', 'lng']]
        dif['dist'] = (dif['lat'] ** 2 + dif['lng'] ** 2) ** 0.5
        df.at[i, 'nearest'] = dif[dif['dist'] == dif['dist'].min()].index[0]

    for col in to_fill_list:
        df.loc[df[df[col].isna() == True].index.tolist(), col] = \
            df.loc[df.loc[df[df[col].isna() == True].index.tolist(), 'nearest'].values, col].values

    print(df[df.isna() == True].sum(axis=1).unique()[0] == 0)

    return df.loc[:, :'price_type']


def get_clusters(data, k):
    X = data[["lat","lng"]]
    best_k = select_best_k(X, k)
    model = KMeans(n_clusters=best_k, init='k-means++')
    
    train_X = X.copy()
    train_X["cluster"] = model.fit_predict(X)

    closest, distances = scipy.cluster.vq.vq(model.cluster_centers_, 
                         train_X.drop("cluster", axis=1).values)

    train_X["centroids"] = 0

    for i in closest:
        train_X["centroids"].iloc[i] = 1

    data[["cluster","centroids"]] = train_X[["cluster","centroids"]]
    
    return data

def select_best_k(data, k):
    X = data[["lat","lng"]]
    max_k = k

    distortions = []

    for i in range(1, max_k+1):
        if len(X) >= i:
            model = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
            model.fit(X)
            distortions.append(model.inertia_)

    k = [i*100 for i in np.diff(distortions,2)].index(min([i*100 for i 
         in np.diff(distortions,2)]))
    
    return k

def sample_by(filters):
    pass
    
def concat_data(frames):
    # get first frame max cluster
    new_frames = []
    new_frames.append(frames[0])
    max_cluster_id = frames[0]["cluster"].max()
    
    # update cluster id's
    for frame in frames[1:]:
        cluster_ids = sorted(frame["cluster"].unique())
        new_cluster_range = [i for i in range(max_cluster_id + 1, max_cluster_id + len(cluster_ids) + 1)]
        remap = {k:v for k, v in zip(cluster_ids, new_cluster_range)}
        frame["cluster"] = frame["cluster"].map(remap).copy()
        new_frames.append(frame)
        max_cluster_id = frame["cluster"].max()
        
    return pd.concat(new_frames)

def plot_clusters(data):
    fig, ax = plt.subplots()
    sns.scatterplot(x="lat", y="lng", data=train, 
                    palette=sns.color_palette("bright",k),
                    hue='cluster', size="centroids", size_order=[1,0],
                    legend="brief", ax=ax).set_title("Clustering (k=" +str(k) + ")")
    th_centroids = model.cluster_centers_
    ax.scatter(th_centroids[:,0], th_centroids[:,1], s=50, c='black', 
               marker="x")