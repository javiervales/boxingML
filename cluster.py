from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import pickle
import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras import regularizers
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D, Conv1D, MaxPooling1D, AveragePooling1D, UpSampling1D, Dropout
from keras.models import Model

def plotClusterEsquema(data,d,y,archivobase,ext="CLUSTER"):
        fig, ax = plt.subplots()
        MARKERS = ["s", "o", "4"]
        COLORS = ["b", "r", "navy", "crimson", "#fdb462", "#7fc97f", "#ff4d4d", "#4d94ff", "#dd66d9", "#5c5c8a"]
        CLASES = [0, 1]

        for clase,marker,color in zip(CLASES, MARKERS, COLORS):
            datatipo = data[d==clase]
            # print(f'\nSCATTER PLOT, tipo {clase}, {datatipo.shape}, {color}, {marker}')
            img = ax.scatter(datatipo[:,0], datatipo[:,1], s=30, color=color, marker=marker, alpha=0.9)

        # Marcamos datos con error
#         datatipo = data[(d!=y) & (y<2)]
#         img = ax.scatter(datatipo[:,0], datatipo[:,1], s=50, color=COLORS[2], marker=MARKERS[2], alpha=0.9, facecolors='none', label='Classification error')
#
#         # Marcamos golpes falsos
#         datatipo = data[(y>2) & (y!=7) & (y!=8)]
#         img = ax.scatter(datatipo[:,0], datatipo[:,1], s=50, color=COLORS[3], marker=MARKERS[2], alpha=0.9, facecolors='none', label='Detection error')
#
#         # Marcamos golpes falsoos
#         datatipo = data[(y==7)]
#         img = ax.scatter(datatipo[:,0], datatipo[:,1], s=50, color=COLORS[4], marker=MARKERS[2], alpha=0.9, facecolors='none', label='Wall hit')
#
#         # Marcamos golpes falsoos
#         datatipo = data[(y==8)]
#         img = ax.scatter(datatipo[:,0], datatipo[:,1], s=100, color=COLORS[5], marker=MARKERS[2], alpha=0.9, facecolors='none', label='Other bag contact')

        ax.set_xticks([])
        ax.set_yticks([])
        #plt.axis('off')
        #ax.legend(fontsize=8)

        fig.savefig(f'{archivobase}/{archivobase}-{ext}.eps', dpi=300)   # save the figure to file
        fig.savefig(f'{archivobase}/{archivobase}-{ext}.png', dpi=300)   # save the figure to file
        fig.savefig(f'{archivobase}/{archivobase}-{ext}.pdf', dpi=300)   # save the figure to file
        plt.close()


def plotCluster(data,d,y,archivobase,ext="CLUSTER"):
        fig, ax = plt.subplots()
        MARKERS = ["s", "o", "o"]
        COLORS = ["b", "r", "#fdb462", "#7fc97f", "#ff4d4d", "#4d94ff", "#dd66d9", "#5c5c8a"]
        CLASES = [0, 1]

        for clase,marker,color in zip(CLASES, MARKERS, COLORS):
            datatipo = data[d==clase]
            # print(f'\nSCATTER PLOT, tipo {clase}, {datatipo.shape}, {color}, {marker}')
            img = ax.scatter(datatipo[:,0], datatipo[:,1], s=30, color=color, marker=marker, alpha=1.0)

        # Marcamos datos con error
        datatipo = data[(d!=y) & (y<2)]
        img = ax.scatter(datatipo[:,0], datatipo[:,1], s=120, color=COLORS[2], marker=MARKERS[2], alpha=0.5, label='Classification error')

        # Marcamos golpes falsos
        datatipo = data[(y>2) & (y!=7) & (y!=8)]
        img = ax.scatter(datatipo[:,0], datatipo[:,1], s=120, color=COLORS[3], marker=MARKERS[2], alpha=0.5, label='Detection error')

        # Marcamos golpes falsoos
        datatipo = data[(y==7)]
        img = ax.scatter(datatipo[:,0], datatipo[:,1], s=120, color=COLORS[4], marker=MARKERS[2], alpha=0.5, label='Wall hit')

        # Marcamos golpes falsoos
        datatipo = data[(y==8)]
        img = ax.scatter(datatipo[:,0], datatipo[:,1], s=120, color=COLORS[5], marker=MARKERS[2], alpha=0.5, label='Other bag contact')

        ax.set_xticks([])
        ax.set_yticks([])
        #plt.axis('off')
        ax.legend(fontsize=8)

        #fig.savefig(f'{archivobase}/{archivobase}-{ext}.png', dpi=300)   # save the figure to file
        fig.savefig(f'{archivobase}/{archivobase}-{ext}.eps', dpi=300)   # save the figure to file
        fig.savefig(f'{archivobase}/{archivobase}-{ext}.pdf', dpi=300)   # save the figure to file
        plt.close()

def matrizP(s):
    P = np.zeros(shape=[3,3])

    for i in range(1,len(s)):
        P[s[i-1]+1,s[i]+1] += 1

    sum_of_rows = P.sum(axis=1)
    np.seterr(divide='ignore', invalid='ignore')
    P = P/sum_of_rows[:, np.newaxis]
    P = np.nan_to_num(P)

    return P


def entropia(y,numgolpessec):
    sec = []
    secnum = 0
    cont = 0
    # Creamos sec
    for i in numgolpessec:
        sec.append([])
        for j in range(i):
            sec[secnum].append(int(y[cont]))
            cont += 1
        # print(f'sec[secnum]: {sec[secnum]}')
        secnum += 1

    #print(sec,cont,secnum)
    P = cont/10 * np.ones([3,3])
    P[0,0] = 0
    P[0,1] = 1
    P[0,2] = 1

    for n in range(secnum):
        s = sec[n].copy()
        s.insert(0,int(-1))
        s.append(int(-1))
        # print(f'{s[1:-1]}',end=',')
        for i in range(1,len(s)):
            #print(f'i: {i}, P[{s[i-1]+1},{s[i]+1}]')
            P[s[i-1]+1,s[i]+1] += 1

    sum_of_rows = P.sum(axis=1)
    P = P/sum_of_rows[:, np.newaxis]

    # print(P)

    Q = np.block([[P.T-np.eye(3)],[np.ones((1,3))]])
    mu = np.matmul(np.matmul(np.linalg.inv(np.matmul(Q.T,Q)),Q.T),np.array([0,0,0,1]))
    #print(P,mu)

    # Calculo de la entropia
    H = 0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i,j]>0:
                H -= mu[i]*P[i,j]*np.log2(P[i,j])
    H11 = H/mu[0]
    #print(f'Entropia: {H11:0.2f}')
    return H11, P

def cluster(X,y,numgolpessec,archivobase,REP=10):
    transformer = preprocessing.MinMaxScaler().fit(X)
    X_transform = transformer.transform(X)

    X_transformsave = X_transform
    Xsave = X
    ysave = y
    listac = []
    listacppal = []
    H = []
    ensemble = np.zeros(y.shape)

    for repeticion in range(REP):
        X_transform = X_transformsave
        X = Xsave
        y = ysave
        significativas = np.ones(y.shape)
        if repeticion>REP/20:
            # Usamos solamente las muestras significativas
            for i in range(significativas.shape[0]):
                if ensemble[i]/repeticion<0.8 and ensemble[i]/repeticion>0.2:             # Muestra con dudas
                    significativas[i] = 0

        print(f'Porcentaje significativas: {significativas.mean()}')
        print(f'significativas: {significativas}')
        overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 100)
        encoder = keras.models.Sequential([keras.layers.Dense(4, input_shape=[X.shape[1]])])
        decoder = keras.models.Sequential([keras.layers.Dense(X.shape[1], input_shape=[4])])
        autoencoder = keras.models.Sequential([encoder, decoder])
        autoencoder.compile(loss="mse", optimizer='adam')
        history = autoencoder.fit(X_transform[significativas==1], X_transform[significativas==1], epochs=20000, callbacks=[overfitCallback], verbose=0)
        loss = history.history['loss'][-1]

        Kmin = 2
        GMM = GaussianMixture(n_components=Kmin).fit(encoder.predict(transformer.transform(X[significativas==1])))
        d = GMM.predict(encoder.predict(transformer.transform(X)))
        y = y
        Haux,_ = entropia(d,numgolpessec)
        H.append(Haux)

        # Visualizamos el resultado
        if repeticion<10:
            data = TSNE(n_components=2).fit_transform(encoder.predict(transformer.transform(X)))
            plotCluster(data,d,y,archivobase)
            data = TSNE(n_components=2).fit_transform(transformer.transform(X))
            plotCluster(data,d,y,archivobase,ext="CLUSTERORIGINAL")

        # Actualizamos el ensemble, d[0] debe ser 0 siempre
        if repeticion==0:
            inicial = d
            ensemble = d
        else:
            # Decidimos si dar o no la vuelta al resultado
            dd = d.copy()
            coincidencias = 0
            for i in range(dd.shape[0]):
                if inicial[i]==dd[i]:
                    coincidencias += 1

            print(f'Porcentaje coincidencias: {coincidencias/dd.shape[0]:0.2f}')
            if coincidencias<dd.shape[0]/2:
            # Damos la vuelta al vector
                for i in range(dd.shape[0]):
                    dd[i]=1 if dd[i]==0 else 0

            # print(f'{ensemble}+{dd}')
            # Actualizamos el ensemble
            ensemble = ensemble + dd

            # Actualizamos inicial
            for i in range(inicial.shape[0]):
                inicial[1] = 1 if ensemble[i]/repeticion>0.5 else 0

        print(f'{ensemble}')

        #print(y)
        #print(d)
        #print(y==d)
        ac = sum(y == d)/d.shape[0]
        if ac<1/Kmin:
            ac = 1-ac
        print(f'Ratio total con aprendizaje no supervisado {100*ac:0.2f}%, loss {loss:0.4f}, H: {H[-1]}')
        listac.append(ac)

        d = GMM.predict(encoder.predict(transformer.transform(X[y<2])))
        yy = y[y<2]
        #print(yy)
        #print(d)
        #print(yy==d)
        acppal = sum(yy == d)/d.shape[0]
        if acppal<1/Kmin:
            acppal = 1-acppal
        print(f'Ratio principal con aprendizaje no supervisado {100*acppal:0.2f}%, loss {loss:0.4f}')
        listacppal.append(acppal)

    ac = np.array(listac)
    acppal = np.array(listacppal)
    H = np.array(H)
    p = ensemble/REP
    dfinal = np.zeros(ensemble.shape)
    for i in range(ensemble.shape[0]):
       dfinal[i] = 0 if p[i]<0.5 else 1
    Hfinal,P = entropia(dfinal, numgolpessec)
    print(f'Pfinal: {P}')

    acfinal = sum(y == dfinal)/dfinal.shape[0]
    print(y==dfinal)
    if acfinal<1/Kmin:
            acfinal = 1-acfinal
    print(f'Ratio final total con aprendizaje no supervisado {100*acfinal:0.2f}%')

    dfinaltotal = dfinal
    dfinal = dfinal[y<2]
    yy = y[y<2]
    acppalfinal = sum(yy == dfinal)/dfinal.shape[0]
    if acppalfinal<1/Kmin:
        acppalfinal = 1-acppalfinal
    print(f'Ratio principal final con aprendizaje no supervisado {100*acppalfinal:0.2f}%')

    print(f'Ratio medio total con aprendizaje no supervisado {100*ac.mean():0.2f}%, {ac.std():0.5f}')
    print(f'Ratio medio principal con aprendizaje no supervisado {100*acppal.mean():0.2f}%, {acppal.std():0.5f}')
    print(f'Entropia media {H.mean():0.2f}, std: {H.std()}')
    print(f'p: {p}')
    print(f'Hfinal: {Hfinal}')
    outliers = 1-significativas.mean()
    print(f'Outliers: {outliers:0.2f}')

    return acfinal, acppalfinal, Hfinal, outliers, dfinaltotal, p


def clusterVAE(X,y,numgolpessec,archivobase,REP=10):
    transformer = preprocessing.MinMaxScaler().fit(X)
    X_transform = transformer.transform(X)

    X_transformsave = X_transform
    Xsave = X
    ysave = y
    listac = []
    listacppal = []
    H = []
    ensemble = np.zeros(y.shape)

    for repeticion in range(REP):
        X_transform = X_transformsave
        X = Xsave
        y = ysave
        significativas = np.ones(y.shape)
        if repeticion>REP/20:
            # Usamos solamente las muestras significativas
            for i in range(significativas.shape[0]):
                if ensemble[i]/repeticion<0.0 and ensemble[i]/repeticion>1.0:             # Muestra con dudas
                    significativas[i] = 0

        print(f'Porcentaje significativas: {significativas.mean()}')
        print(f'significativas: {significativas}')

        '''  ------------------------------------------------------------------------------
                                             GET NETWORK PARAMS
            ------------------------------------------------------------------------------ '''
        from bunch import Bunch
        import utils.utils as utils
        import utils.constants as const 
        print(X.shape)
        network_params = Bunch()
        network_params.input_height = 1
        network_params.input_width = X.shape[1]
        network_params.input_nchannels = 1
        
        network_params.hidden_dim = 4 
        network_params.z_dim = 1 
        network_params.w_dim = 1 
        network_params.K =  2
        network_params.num_layers = 1
        
        '''  -----------------------------------------------------------------------------
                                COMPUTATION GRAPH (Build the model)
            ------------------------------------------------------------------------------ '''
        from GMVAE_model import GMVAEModel
        vae_model = GMVAEModel(network_params,sigma=0.001, sigma_act=utils.softplus_bias,
                               transfer_fct=tf.nn.relu,learning_rate=0.1,
                               kinit=tf.contrib.layers.xavier_initializer(),
                               batch_size=X_transform.shape[0], drop_rate=0.001, 
                               epochs=2000, checkpoint_dir='~/.checkpoint', 
                               summary_dir='~/.checkpoint', result_dir='~/.checkpoint', 
                               restore=0, model_type=2)
        print('\nNumber of trainable paramters', vae_model.trainable_count)
        
        '''  -----------------------------------------------------------------------------
                                TRAIN THE MODEL
            ------------------------------------------------------------------------------ '''
        from utils.dataset import Dataset
        data_train = X_transform[significativas==1].reshape([-1,X.shape[1],1,1])
        XXX = Dataset(data_train, np.zeros(X_transform[significativas==1].shape[0]))
        XALL = Dataset(X_transform.reshape([-1,X.shape[1],1,1]), np.zeros(X_transform.shape[0]))
        vae_model.train(XXX, XXX)
    
        x_input, x_labels, x_recons, z_recons, w_recons, y_recons = vae_model.reconstruct_input_all(XALL)
        print(y_recons)

        Kmin = 2
        d = np.argmax(y_recons,axis=1)
        Haux,_ = entropia(d,numgolpessec)
        H.append(Haux)

        # Visualizamos el resultado
        # Actualizamos el ensemble, d[0] debe ser 0 siempre
        if repeticion==0:
            inicial = d
            ensemble = d
        else:
            # Decidimos si dar o no la vuelta al resultado
            dd = d.copy()
            coincidencias = 0
            for i in range(dd.shape[0]):
                if inicial[i]==dd[i]:
                    coincidencias += 1

            print(f'Porcentaje coincidencias: {coincidencias/dd.shape[0]:0.2f}')
            if coincidencias<dd.shape[0]/2:
            # Damos la vuelta al vector
                for i in range(dd.shape[0]):
                    dd[i]=1 if dd[i]==0 else 0

            # print(f'{ensemble}+{dd}')
            # Actualizamos el ensemble
            ensemble = ensemble + dd

            # Actualizamos inicial
            for i in range(inicial.shape[0]):
                inicial[1] = 1 if ensemble[i]/repeticion>0.5 else 0

        print(f'{ensemble}')

        #print(y)
        #print(d)
        #print(y==d)
        ac = sum(y == d)/d.shape[0]
        if ac<1/Kmin:
            ac = 1-ac
        print(f'Ratio total con aprendizaje no supervisado {100*ac:0.2f}%, H: {H[-1]}')
        listac.append(ac)

        d = d[y<2]
        yy = y[y<2]
        #print(yy)
        #print(d)
        #print(yy==d)
        acppal = sum(yy == d)/d.shape[0]
        if acppal<1/Kmin:
            acppal = 1-acppal
        print(f'Ratio principal con aprendizaje no supervisado {100*acppal:0.2f}%')
        listacppal.append(acppal)

    ac = np.array(listac)
    acppal = np.array(listacppal)
    H = np.array(H)
    p = ensemble/REP
    dfinal = np.zeros(ensemble.shape)
    for i in range(ensemble.shape[0]):
       dfinal[i] = 0 if p[i]<0.5 else 1
    Hfinal,P = entropia(dfinal, numgolpessec)
    print(f'Pfinal: {P}')

    acfinal = sum(y == dfinal)/dfinal.shape[0]
    print(y==dfinal)
    if acfinal<1/Kmin:
            acfinal = 1-acfinal
    print(f'Ratio final total con aprendizaje no supervisado {100*acfinal:0.2f}%')

    dfinaltotal = dfinal
    dfinal = dfinal[y<2]
    yy = y[y<2]
    acppalfinal = sum(yy == dfinal)/dfinal.shape[0]
    if acppalfinal<1/Kmin:
        acppalfinal = 1-acppalfinal
    print(f'Ratio principal final con aprendizaje no supervisado {100*acppalfinal:0.2f}%')

    print(f'Ratio medio total con aprendizaje no supervisado {100*ac.mean():0.2f}%, {ac.std():0.5f}')
    print(f'Ratio medio principal con aprendizaje no supervisado {100*acppal.mean():0.2f}%, {acppal.std():0.5f}')
    print(f'Entropia media {H.mean():0.2f}, std: {H.std()}')
    print(f'p: {p}')
    print(f'Hfinal: {Hfinal}')
    outliers = 1-significativas.mean()
    print(f'Outliers: {outliers:0.2f}')

    return acfinal, acppalfinal, Hfinal, outliers, dfinaltotal, p


def clusterPCA(X,y,numgolpessec,archivobase,REP=10):
    transformer = preprocessing.MinMaxScaler().fit(X)
    X_transform = transformer.transform(X)

    X_transformsave = X_transform
    Xsave = X
    ysave = y
    listac = []
    listacppal = []
    H = []
    ensemble = np.zeros(y.shape)

    for repeticion in range(REP):
        X_transform = X_transformsave
        X = Xsave
        y = ysave
        significativas = np.ones(y.shape)
        if repeticion>REP/20:
            # Usamos solamente las muestras significativas
            for i in range(significativas.shape[0]):
                if ensemble[i]/repeticion<0.8 and ensemble[i]/repeticion>0.2:             # Muestra con dudas
                    significativas[i] = 0

        print(f'Porcentaje significativas: {significativas.mean()}')
        print(f'significativas: {significativas}')
        pca = PCA(n_components=2)
        pca.fit(X_transform[significativas==1])

        Kmin = 2
        GMM = GaussianMixture(n_components=Kmin).fit(pca.transform(transformer.transform(X[significativas==1])))
        d = GMM.predict(pca.transform(transformer.transform(X)))
        y = y
        Haux,_ = entropia(d,numgolpessec)
        H.append(Haux)

        # Visualizamos el resultado
        if repeticion<10:
        #    data = TSNE(n_components=2).fit_transform(encoder.predict(transformer.transform(X)))
        #    plotCluster(data,d,y,archivobase)
            data = TSNE(n_components=2).fit_transform(transformer.transform(X))
            plotCluster(data,d,y,archivobase,ext=f'rep{repeticion}')

        # Actualizamos el ensemble, d[0] debe ser 0 siempre
        if repeticion==0:
            inicial = d
            ensemble = d
        else:
            # Decidimos si dar o no la vuelta al resultado
            dd = d.copy()
            coincidencias = 0
            for i in range(dd.shape[0]):
                if inicial[i]==dd[i]:
                    coincidencias += 1

            print(f'Porcentaje coincidencias: {coincidencias/dd.shape[0]:0.2f}')
            if coincidencias<dd.shape[0]/2:
            # Damos la vuelta al vector
                for i in range(dd.shape[0]):
                    dd[i]=1 if dd[i]==0 else 0

            # print(f'{ensemble}+{dd}')
            # Actualizamos el ensemble
            ensemble = ensemble + dd

            # Actualizamos inicial
            for i in range(inicial.shape[0]):
                inicial[1] = 1 if ensemble[i]/repeticion>0.5 else 0

        print(f'{ensemble}')

        #print(y)
        #print(d)
        #print(y==d)
        ac = sum(y == d)/d.shape[0]
        if ac<1/Kmin:
            ac = 1-ac
        print(f'Ratio total con aprendizaje no supervisado {100*ac:0.2f}%, H: {H[-1]}')
        listac.append(ac)

        d = GMM.predict(pca.transform(transformer.transform(X[y<2])))
        yy = y[y<2]
        #print(yy)
        #print(d)
        #print(yy==d)
        acppal = sum(yy == d)/d.shape[0]
        if acppal<1/Kmin:
            acppal = 1-acppal
        print(f'Ratio principal con aprendizaje no supervisado {100*acppal:0.2f}%')
        listacppal.append(acppal)

    ac = np.array(listac)
    acppal = np.array(listacppal)
    H = np.array(H)
    p = ensemble/REP
    dfinal = np.zeros(ensemble.shape)
    for i in range(ensemble.shape[0]):
       dfinal[i] = 0 if p[i]<0.5 else 1
    Hfinal,P = entropia(dfinal, numgolpessec)
    print(f'Pfinal: {P}')

    acfinal = sum(y == dfinal)/dfinal.shape[0]
    print(y==dfinal)
    if acfinal<1/Kmin:
            acfinal = 1-acfinal
    print(f'Ratio final total con aprendizaje no supervisado {100*acfinal:0.2f}%')

    dfinaltotal = dfinal
    dfinal = dfinal[y<2]
    yy = y[y<2]
    acppalfinal = sum(yy == dfinal)/dfinal.shape[0]
    if acppalfinal<1/Kmin:
        acppalfinal = 1-acppalfinal
    print(f'Ratio principal final con aprendizaje no supervisado {100*acppalfinal:0.2f}%')

    print(f'Ratio medio total con aprendizaje no supervisado {100*ac.mean():0.2f}%, {ac.std():0.5f}')
    print(f'Ratio medio principal con aprendizaje no supervisado {100*acppal.mean():0.2f}%, {acppal.std():0.5f}')
    print(f'Entropia media {H.mean():0.2f}, std: {H.std()}')
    print(f'p: {p}')
    print(f'Hfinal: {Hfinal}')
    outliers = 1-significativas.mean()
    print(f'Outliers: {outliers:0.2f}')

    return acfinal, acppalfinal, Hfinal, outliers, dfinaltotal, p


def clusterSINAUTOENC(X,y,numgolpessec,archivobase,REP=10):
    transformer = preprocessing.MinMaxScaler().fit(X)
    X_transform = transformer.transform(X)

    X_transformsave = X_transform
    Xsave = X
    ysave = y
    listac = []
    listacppal = []
    H = []
    ensemble = np.zeros(y.shape)

    for repeticion in range(REP):
        X_transform = X_transformsave
        X = Xsave
        y = ysave
        significativas = np.ones(y.shape)
        if repeticion>REP/20:
            # Usamos solamente las muestras significativas
            for i in range(significativas.shape[0]):
                if ensemble[i]/repeticion<0.8 and ensemble[i]/repeticion>0.2:             # Muestra con dudas
                    significativas[i] = 0

        print(f'Porcentaje significativas: {significativas.mean()}')
        print(f'significativas: {significativas}')

        Kmin = 2
        GMM = GaussianMixture(n_components=Kmin).fit(transformer.transform(X[significativas==1]))
        d = GMM.predict(transformer.transform(X))
        y = y
        Haux,_ = entropia(d,numgolpessec)
        H.append(Haux)

        # Visualizamos el resultado
#        if repeticion<10:
#            data = TSNE(n_components=2).fit_transform(encoder.predict(transformer.transform(X)))
#            plotCluster(data,d,y,archivobase)
#            data = TSNE(n_components=2).fit_transform(transformer.transform(X))
#            plotCluster(data,d,y,archivobase,ext="CLUSTERORIGINAL")

        # Actualizamos el ensemble, d[0] debe ser 0 siempre
        if repeticion==0:
            inicial = d
            ensemble = d
        else:
            # Decidimos si dar o no la vuelta al resultado
            dd = d.copy()
            coincidencias = 0
            for i in range(dd.shape[0]):
                if inicial[i]==dd[i]:
                    coincidencias += 1

            print(f'Porcentaje coincidencias: {coincidencias/dd.shape[0]:0.2f}')
            if coincidencias<dd.shape[0]/2:
            # Damos la vuelta al vector
                for i in range(dd.shape[0]):
                    dd[i]=1 if dd[i]==0 else 0

            # print(f'{ensemble}+{dd}')
            # Actualizamos el ensemble
            ensemble = ensemble + dd

            # Actualizamos inicial
            for i in range(inicial.shape[0]):
                inicial[1] = 1 if ensemble[i]/repeticion>0.5 else 0

        print(f'{ensemble}')

        #print(y)
        #print(d)
        #print(y==d)
        ac = sum(y == d)/d.shape[0]
        if ac<1/Kmin:
            ac = 1-ac
        print(f'Ratio total con aprendizaje no supervisado {100*ac:0.2f}%, H: {H[-1]}')
        listac.append(ac)

        d = GMM.predict(transformer.transform(X[y<2]))
        yy = y[y<2]
        #print(yy)
        #print(d)
        #print(yy==d)
        acppal = sum(yy == d)/d.shape[0]
        if acppal<1/Kmin:
            acppal = 1-acppal
        print(f'Ratio principal con aprendizaje no supervisado {100*acppal:0.2f}%')
        listacppal.append(acppal)

    ac = np.array(listac)
    acppal = np.array(listacppal)
    H = np.array(H)
    p = ensemble/REP
    dfinal = np.zeros(ensemble.shape)
    for i in range(ensemble.shape[0]):
       dfinal[i] = 0 if p[i]<0.5 else 1
    Hfinal,P = entropia(dfinal, numgolpessec)
    print(f'Pfinal: {P}')

    acfinal = sum(y == dfinal)/dfinal.shape[0]
    print(y==dfinal)
    if acfinal<1/Kmin:
            acfinal = 1-acfinal
    print(f'Ratio final total con aprendizaje no supervisado {100*acfinal:0.2f}%')

    dfinaltotal = dfinal
    dfinal = dfinal[y<2]
    yy = y[y<2]
    acppalfinal = sum(yy == dfinal)/dfinal.shape[0]
    if acppalfinal<1/Kmin:
        acppalfinal = 1-acppalfinal
    print(f'Ratio principal final con aprendizaje no supervisado {100*acppalfinal:0.2f}%')

    print(f'Ratio medio total con aprendizaje no supervisado {100*ac.mean():0.2f}%, {ac.std():0.5f}')
    print(f'Ratio medio principal con aprendizaje no supervisado {100*acppal.mean():0.2f}%, {acppal.std():0.5f}')
    print(f'Entropia media {H.mean():0.2f}, std: {H.std()}')
    print(f'p: {p}')
    print(f'Hfinal: {Hfinal}')
    outliers = 1-significativas.mean()
    print(f'Outliers: {outliers:0.2f}')

    return acfinal, acppalfinal, Hfinal, outliers, dfinaltotal, p


def clusterDL(X,y,numgolpessec,archivobase,REP=10):

    newX = []
    for i in range(X.shape[0]):
        muestraX = np.reshape(X[i], (-1,5))
        newX.append(muestraX.transpose())
    X = np.array(newX)

    Xsave = X
    ysave = y
    listac = []
    listacppal = []
    H = []
    ensemble = np.zeros(y.shape)

    for repeticion in range(REP):
        X = Xsave
        y = ysave
        significativas = np.ones(y.shape)
        if repeticion>REP/20:
            # Usamos solamente las muestras significativas
            for i in range(significativas.shape[0]):
                if ensemble[i]/repeticion<0.8 and ensemble[i]/repeticion>0.2:             # Muestra con dudas
                    significativas[i] = 0

        print(f'Porcentaje significativas: {significativas.mean()}')
        print(f'significativas: {significativas}')
        overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 100)

        print(X.shape)

        encoder = keras.models.Sequential([
		    Conv1D(8,3,activation='relu', padding='same',input_shape=[X.shape[1],X.shape[2]]),
		    Conv1D(16,3,activation='relu', padding='same'),
		    Dropout(0.2),
			MaxPooling1D(pool_size=2),
			Flatten(),
			Dense(2,activation='softmax')
		])

        decoder = keras.models.Sequential([
			Dense(X.shape[1]*X.shape[2],activation='relu'),
		    Dropout(0.2),
			Reshape((X.shape[1],X.shape[2])),
		    Conv1D(10,3,activation='relu', padding='same'),
		    Conv1D(X.shape[2],3,activation='relu', padding='same'),
		])

        autoencoder = keras.models.Sequential([encoder, decoder])
        encoder.summary()
        decoder.summary()
        autoencoder.summary()

        autoencoder.compile(loss="mse", optimizer='adam')
        history = autoencoder.fit(X[significativas==1], X[significativas==1], epochs=20000, callbacks=[overfitCallback], verbose=0)
        loss = history.history['loss'][-1]

        Kmin = 2
        GMM = GaussianMixture(n_components=Kmin).fit(encoder.predict(X[significativas==1]))
        d = GMM.predict(encoder.predict(X))
        y = y
        Haux,_ = entropia(d,numgolpessec)
        H.append(Haux)

        # Visualizamos el resultado
        #if repeticion<10:
        #    data = TSNE(n_components=2).fit_transform(encoder.predict(transformer.transform(X)))
        #    plotCluster(data,d,y,archivobase)
        #    data = TSNE(n_components=2).fit_transform(transformer.transform(X))
        #    plotCluster(data,d,y,archivobase,ext="CLUSTERORIGINAL")

        # Actualizamos el ensemble, d[0] debe ser 0 siempre
        if repeticion==0:
            inicial = d
            ensemble = d
        else:
            # Decidimos si dar o no la vuelta al resultado
            dd = d.copy()
            coincidencias = 0
            for i in range(dd.shape[0]):
                if inicial[i]==dd[i]:
                    coincidencias += 1

            print(f'Porcentaje coincidencias: {coincidencias/dd.shape[0]:0.2f}')
            if coincidencias<dd.shape[0]/2:
            # Damos la vuelta al vector
                for i in range(dd.shape[0]):
                    dd[i]=1 if dd[i]==0 else 0

            # print(f'{ensemble}+{dd}')
            # Actualizamos el ensemble
            ensemble = ensemble + dd

            # Actualizamos inicial
            for i in range(inicial.shape[0]):
                inicial[1] = 1 if ensemble[i]/repeticion>0.5 else 0

        print(f'{ensemble}')

        #print(y)
        #print(d)
        #print(y==d)
        ac = sum(y == d)/d.shape[0]
        if ac<1/Kmin:
            ac = 1-ac
        print(f'Ratio total con aprendizaje no supervisado {100*ac:0.2f}%, loss {loss:0.4f}, H: {H[-1]}')
        listac.append(ac)

        d = GMM.predict(encoder.predict(X[y<2]))
        yy = y[y<2]
        #print(yy)
        #print(d)
        #print(yy==d)
        acppal = sum(yy == d)/d.shape[0]
        if acppal<1/Kmin:
            acppal = 1-acppal
        print(f'Ratio principal con aprendizaje no supervisado {100*acppal:0.2f}%, loss {loss:0.4f}')
        listacppal.append(acppal)

    ac = np.array(listac)
    acppal = np.array(listacppal)
    H = np.array(H)
    p = ensemble/REP
    dfinal = np.zeros(ensemble.shape)
    for i in range(ensemble.shape[0]):
       dfinal[i] = 0 if p[i]<0.5 else 1
    Hfinal,P = entropia(dfinal, numgolpessec)
    print(f'Pfinal: {P}')

    acfinal = sum(y == dfinal)/dfinal.shape[0]
    print(y==dfinal)
    if acfinal<1/Kmin:
            acfinal = 1-acfinal
    print(f'Ratio final total con aprendizaje no supervisado {100*acfinal:0.2f}%')

    dfinaltotal = dfinal
    dfinal = dfinal[y<2]
    yy = y[y<2]
    acppalfinal = sum(yy == dfinal)/dfinal.shape[0]
    if acppalfinal<1/Kmin:
        acppalfinal = 1-acppalfinal
    print(f'Ratio principal final con aprendizaje no supervisado {100*acppalfinal:0.2f}%')

    print(f'Ratio medio total con aprendizaje no supervisado {100*ac.mean():0.2f}%, {ac.std():0.5f}')
    print(f'Ratio medio principal con aprendizaje no supervisado {100*acppal.mean():0.2f}%, {acppal.std():0.5f}')
    print(f'Entropia media {H.mean():0.2f}, std: {H.std()}')
    print(f'p: {p}')
    print(f'Hfinal: {Hfinal}')
    outliers = 1-significativas.mean()
    print(f'Outliers: {outliers:0.2f}')

    return acfinal, acppalfinal, Hfinal, outliers, dfinaltotal, p



