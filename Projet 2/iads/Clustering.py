import numpy as np
import matplotlib.pyplot as plt
import copy

def normalisation(DataFrame):
    df = DataFrame.copy()
    for column in DataFrame.columns: 
        df[column] = (df[column] - df[column].min()) /( df[column].max() - df[column].min())
    return(df)

def dist_euclidienne(vec1,vec2):
    return np.linalg.norm(vec1 - vec2)

def dist_manhattan(vec1, vec2):
    res = 0
    for i in range(len(vec1)):
        res += abs(vec1[i] - vec2[i])
    return res

def dist_vect(nom, vec1, vec2):
    if nom == "euclidienne":
        return dist_euclidienne(vec1, vec2)
    elif nom == "manhattan":
        return dist_manhattan(vec1,vec2)
    else:
        print("Distance :", nom, " non acceptée.")

def centroide(DataFrame):
    data1 = np.array(DataFrame)
    res = np.mean(data1[:,-2:], axis=0)
    return res

def dist_centroides(vec1, vec2):
    return dist_euclidienne(centroide(vec1),centroide (vec2))

def initialise(DataFrame):
    dic = dict()
    x,y = DataFrame.shape
    for i in range(0,x):
        dic.update({i: [i]})
    return dic

def fusionne(DF, PO,verbose = False):
    indice = (0,0)
    min = 10000
    toRead = PO.keys()
    for i in toRead:
        for j in toRead:
            if i == j:
                continue
            dist = dist_centroides(DF.iloc[PO[i]],DF.iloc[PO[j]])
            if dist<= min :
                min = dist
                indice = (i,j)
    if verbose == True :
        print("Distance mininimale trouvée entre : [",i,j,"] = ",min)
    size = max(PO.keys())
    i,j = indice
    P1 = copy.deepcopy(PO)
    P1.pop(i)
    P1.pop(j)
    P1[size+1] = PO[i]+PO[j]
    return P1,i,j,min

def clustering_hierarchique(DF, verbose, dendrogramme) :
    PO = initialise(DF)
    liste = []
    val = 0
    tmp = copy.deepcopy(PO)
    while (val != len(DF)):
        P1,i,j,dist = fusionne(DF,tmp, verbose)
        val = len(tmp[i])+len(tmp[j])
        liste.append([i,j,dist,val])
        tmp = P1
    if (dendrogramme == False):
        return liste
    import scipy.cluster.hierarchy

    # Paramètre de la fenêtre d'affichage: 
    plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
    plt.title('Dendrogramme', fontsize=25)    
    plt.xlabel("Indice d'exemple", fontsize=25)
    plt.ylabel('Distance', fontsize=25)

    # Construction du dendrogramme pour notre clustering :
    scipy.cluster.hierarchy.dendrogram(
        liste, 

        leaf_font_size=24.,  # taille des caractères de l'axe des X
    )

    # Affichage du résultat obtenu:
    plt.show()
    
    return liste