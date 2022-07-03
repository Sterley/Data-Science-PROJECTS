# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""


import numpy as np
import matplotlib.pyplot as plt
import math

# plot2DSet:
def plot2DSet(array1, array2):
    # Extraction des exemples de classe -1:
    data2_negatifs2 = array1[array2 == -1]
    # Extraction des exemples de classe +1:
    data2_positifs2 = array1[array2 == +1]
    plt.scatter(data2_negatifs2[:,0],data2_negatifs2[:,1],marker='o', color="red") # 'o' rouge pour la classe -1
    plt.scatter(data2_positifs2[:,0],data2_positifs2[:,1],marker='x', color="blue") # 'x' bleu pour la classe +1
    
# ------------------------ 
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])
	
# genere_dataset_uniform:
def genere_dataset_uniform(n, p, inf, sup):
    np.random.seed(42)  
    data_desc = np.random.uniform(inf,sup,(p,n)) 
    data_label = np.asarray([-1 for i in range(0,int(p/2))] + [+1 for i in range(0,int(p/2))])
    return data_desc, data_label

# genere_dataset_gaussian:
def genere_dataset_gaussian(pc, ps, nc, ns, nbp):
    pos = np.random.multivariate_normal(pc, ps, size=nbp)
    neg = np.random.multivariate_normal(nc, ns, size=nbp)
    tab1 = np.vstack((pos, neg))
    tab2 = np.asarray([-1 for i in range(nbp)] + [+1 for i in range(nbp)])
    return tab1, tab2

# create XOR
def create_XOR(n, sigma):
  one = sigma * np.random.randn(n, 2) + [0, 0]
  two = sigma * np.random.randn(n, 2) + [1, 1]
  three = sigma * np.random.randn(n, 2) + [0, 1]
  four = sigma * np.random.randn(n, 2) + [1, 0]
  return (np.vstack((one, two, three, four)), np.hstack((-1*np.ones(2*n), np.ones(2*n))))


def classe_majoritaire(Y):
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    return valeurs[np.argmax(nb_fois)]


def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    if (len(P)==0 or len(P)==1):
        return 0.0
    somme=0
    for p in P:
        if p != 0:
            somme+= p*math.log(p,len(P))
    return - somme


def shannon2(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    if (len(P)==0 or len(P)==1):
        return 0.0
    somme=0
    for p in P:
        if p != 0:
            somme+= p*math.log(p)
    return - somme


def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    Etiq = []
    dict_etiq = {}
    for lab in Y:
        if lab not in Etiq:
            Etiq.append(lab)
            dict_etiq[lab] = 1
        else:
            tmp = dict_etiq[lab]
            tmp += 1
            dict_etiq[lab] = tmp
    
    P = []
    for etiq in dict_etiq.items():
        P.append(etiq[1]/len(Y))
    return shannon(P)


def entropie2(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    Etiq = []
    dict_etiq = {}
    for lab in Y:
        if lab not in Etiq:
            Etiq.append(lab)
            dict_etiq[lab] = 1
        else:
            tmp = dict_etiq[lab]
            tmp += 1
            dict_etiq[lab] = tmp
    
    P = []
    for etiq in dict_etiq.items():
        P.append(etiq[1]/len(Y))
    return shannon2(P)
