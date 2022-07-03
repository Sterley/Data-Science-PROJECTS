# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2022

# Import de packages externes
import numpy as np
import math
import sys
import random

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


def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            - nb_classes : (int) nombre initial de labels dans le dataset (défaut: 2)
        output: tuple : ((seuil_trouve, entropie), (liste_coupures,liste_entropies))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropie (float): entropie du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropies (List[float]): la liste des entropies correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropie de la coupure
        val_entropie_inf = entropie2(cl_inf) # entropie de l'ensemble des inf
        val_entropie_sup = entropie2(cl_sup) # entropie de l'ensemble des sup
        
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    
    return (best_seuil, best_entropie), (liste_coupures,liste_entropies)


def partitionne(m_desc, m_label, n, s):
    left_data = []
    left_class = []
    right_data = [] 
    right_class = []
    for i in range(len(m_desc)):
        ex = m_desc[i]
        if ex[n] <= s:
            left_data.append(ex)
            left_class.append(m_label[i])
        else :
            right_data.append(ex)
            right_class.append(m_label[i])
    return ((np.array(left_data),np.array(left_class)), (np.array(right_data),np.array(right_class)))

def tirage(VX, m, r):
    if r==True:
        ret = []
        i = 0
        while i < m:
            c = random.choice(VX)
            ret.append(c)
            i +=1
        return ret
    else:
        ret = []
        i = 0
        while i < m:
            c = random.choice(VX)
            while c in ret:
                c = random.choice(VX)
            ret.append(c)
            i +=1
        return ret



            


class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g


class NoeudNumerique:
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.seuil = None          # seuil de coupure pour ce noeud
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()            
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup        
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        #############
        # COMPLETER CETTE PARTIE 
        #
        #############
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.Les_fils['inf'].classifie(exemple)
        if exemple[self.attribut] > self.seuil:
            return self.Les_fils['sup'].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc 
            pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g,prefixe+"g")
            self.Les_fils['sup'].to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))                
        return g



def construit_AD_num(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie2(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudNumerique(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = float('-Inf')  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1               # numéro du meilleur attribut (init à -1 (aucun))
        Xbest_set = None
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_tuple : le tuple rendu par partionne() pour le meilleur attribut trouvé
        # Xbest_seuil : le seuil de partitionnement associé au meilleur attribut
        #
        # Remarque : attention, la fonction discretise() peut renvoyer un tuple contenant
        # None (pas de partitionnement possible)n dans ce cas, on considèrera que le
        # résultat d'un partitionnement est alors ((X,Y),(None,None))
        
        tabHS = []
        # pour chaque attribut  𝑋𝑗  qui décrit les exemples de  𝑋 
        for i in range(len(LNoms)):
            Xj = LNoms[i]
            
            # pour chacune des valeurs 𝑣𝑗𝑙 de 𝑋𝑗
            valeurs_vjl = []
            dict_valeurs_vjl = {}
            for exemp in X:
                if exemp[i] not in valeurs_vjl:
                    valeurs_vjl.append(exemp[i])
                    # construire l'ensemble des exemples de  𝑋  qui possède la valeur  𝑣𝑗𝑙  
                    # ainsi que l'ensemble de leurs labels
                    vjl = exemp[i]
                    exemples_X = []
                    leurs_labels = []
                    for j in range(len(X)):
                        ex = X[j]
                        if vjl in ex:
                            exemples_X.append(ex)
                            leurs_labels.append(Y[j])
                    dict_valeurs_vjl[vjl] = [exemples_X, leurs_labels]
            
            # calculer l'entropie conditionnelle de Shannon de la classe relativement à l'attribut  𝑋𝑗 . 
            # On note  𝐻𝑆(𝑌|𝑋𝑗)  cette entropie.
            HSy_xj = 0
            for vjl in valeurs_vjl:
                #nbocc = nbr de ligne il a apparait
                entro_vjl = entropie2(dict_valeurs_vjl[vjl][1]) * (len(dict_valeurs_vjl[vjl][0])/len(Y))
                HSy_xj = HSy_xj + entro_vjl
            tabHS.append(HSy_xj)
        
        i_best = tabHS.index(min(tabHS))
        gain_max = tabHS[i_best]
        Xbest_valeurs = np.unique(X[:, i_best])

        resultat, liste_vals = discretise(X,Y,i_best)
        if resultat[0]==None or resultat[1]==None or liste_vals[0]==None or liste_vals[1]==None:
            Xbest_seuil = resultat[0]
            Xbest_tuple = ((X,Y),(None,None))
        else:
            Xbest_seuil = resultat[0]
            Xbest_tuple = partitionne(X,Y,i_best,Xbest_seuil)
        
        ############
        if (gain_max != float('-Inf')):
            if len(LNoms)>0:  # si on a des noms de features
                noeud = NoeudNumerique(i_best,LNoms[i_best]) 
            else:
                noeud = NoeudNumerique(i_best)
            ((left_data,left_class), (right_data,right_class)) = Xbest_tuple
            noeud.ajoute_fils( Xbest_seuil, \
                              construit_AD_num(left_data,left_class, epsilon, LNoms), \
                              construit_AD_num(right_data,right_class, epsilon, LNoms) )
        else: # aucun attribut n'a pu améliorer le gain d'information
              # ARRET : on crée une feuille
            noeud = NoeudNumerique(-1,"Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))
        
    return noeud



def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = sys.float_info.min  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1         # numéro du meilleur attribut
        Xbest_valeurs = None
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur du gain d'information pour chaque attribut.
        
        tabHS = []
        # pour chaque attribut  𝑋𝑗  qui décrit les exemples de  𝑋 
        for i in range(len(LNoms)):
            Xj = LNoms[i]
            
            # pour chacune des valeurs 𝑣𝑗𝑙 de 𝑋𝑗
            valeurs_vjl = []
            dict_valeurs_vjl = {}
            for exemp in X:
                if exemp[i] not in valeurs_vjl:
                    valeurs_vjl.append(exemp[i])
                    # construire l'ensemble des exemples de  𝑋  qui possède la valeur  𝑣𝑗𝑙  
                    # ainsi que l'ensemble de leurs labels
                    vjl = exemp[i]
                    exemples_X = []
                    leurs_labels = []
                    for j in range(len(X)):
                        ex = X[j]
                        if vjl in ex:
                            exemples_X.append(ex)
                            leurs_labels.append(Y[j])
                    dict_valeurs_vjl[vjl] = [exemples_X, leurs_labels]
            
            # calculer l'entropie conditionnelle de Shannon de la classe relativement à l'attribut  𝑋𝑗 . 
            # On note  𝐻𝑆(𝑌|𝑋𝑗)  cette entropie.
            HSy_xj = 0
            for vjl in valeurs_vjl:
                #nbocc = nbr de ligne il a apparait
                entro_vjl = entropie(dict_valeurs_vjl[vjl][1]) * (len(dict_valeurs_vjl[vjl][0])/len(Y))
                HSy_xj = HSy_xj + entro_vjl
            tabHS.append(HSy_xj)
        
        preced = tabHS[0]
        for i in range(len(tabHS)):
            if tabHS[i] < preced:
                preced = tabHS[i]
                i_best = i
        gain_max = tabHS[i_best]
        Xbest_valeurs = np.unique(X[:, i_best])
            
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud































    
# ---------------------------
# ------------------------ A COMPLETER :
class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """ 
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # ------------------------------
        # COMPLETER CETTE FONCTION ICI : 
        
        count=0
        for i in range(len(label_set)):
          if self.predict(desc_set[i]) == label_set[i]:
            count+=1
        return count/len(label_set)
        
        # ------------------------------
        
# ---------------------------
# ------------------------ A COMPLETER :

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k=k
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        size = np.linalg.norm(self.desc_set-x, axis=1)
        argsort = np.argsort(size)
        score = 0
        for i in argsort[:self.k]:
          if self.label_set[i] == +1:
            score +=1
        return 2*(score/self.k -.5)
            
    
    def predict(self, x):
        score = self.score(x)
        if score > 0.5:
            return 1
        return -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set


# ------------------------ A COMPLETER :
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        nbDim = input_dimension
        self.w = np.asarray([np.random.uniform() for i in range(nbDim)])
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        print("Pas d'apprentissage pour ce classifieur")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        #dot pour le produit scalaire
        return np.dot(x,self.w)
    
    def predict(self, x):
        score = self.score(x)
        if score > 0.5:
            return 1
        return -1


class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.init = init
        res = []
        if init == 0:
          self.w_de_base = np.asarray(np.zeros(self.input_dimension))
        elif init == 1:
          for i in range(self.input_dimension):
            res.append((2*i - 1)*0.001)
          self.w_de_base = np.asarray(res)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """    
        
        liste_indice = np.random.permutation(len(desc_set))
        for i in liste_indice:
            x=desc_set[i]
            y=label_set[i]
            pred = self.predict(x)
            if y != pred:
                self.w_de_base = self.w_de_base + self.learning_rate*x*y
        return self.w_de_base
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """     

        ret = []  
        i = 1
        w_old = self.train_step(desc_set, label_set)
        while i < niter_max:
          w_new = self.train_step(desc_set, label_set)
          d = np.linalg.norm(w_old - w_new)
          ret.append(d)
          #if i != 1:
            #if d < seuil:
              #break
          w_old = w_new.copy()
          i += 1
        ret.pop(0)
        return ret
    
    def score(self,x):
      score = 0
      for i in range(len(x)):
        score += self.w_de_base[i] * x[i]
      return score
    
    def predict(self, x):
      score = self.score(x)
      return 1 if score >= 0.0 else -1

    



class ClassifierPerceptronBiais(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.init = init
        res = []
        if init == 0:
          self.w_de_base = np.asarray(np.zeros(self.input_dimension))
        elif init == 1:
          for i in range(self.input_dimension):
            res.append((2*i - 1)*0.001)
          self.w_de_base = np.asarray(res)
        self.allw = []
        self.C = []
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """   
        self.desc_set = desc_set
        self.label_set = label_set
        
        liste_indice = np.random.permutation(len(desc_set))
        for i in liste_indice:
            x=desc_set[i]
            y=label_set[i]
            pred = self.predict(x)
            if self.score(x)*y < 1:
                self.w_de_base = self.w_de_base + self.learning_rate*x*(y-self.score(x))
                self.allw.append(self.w_de_base.copy())
                self.C.append(self.cout(desc_set, label_set))
        return self.w_de_base
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """     

        ret = []  
        i = 1
        w_old = self.train_step(desc_set, label_set)
        while i < niter_max:
          w_new = self.train_step(desc_set, label_set)
          d = np.linalg.norm(w_old - w_new)
          ret.append(d)
          #if i != 1:
            #if d < seuil:
              #break
          w_old = w_new.copy()
          i += 1
        ret.pop(0)
        return ret
    
    def score(self,x):
      score = 0
      for i in range(len(x)):
        score += self.w_de_base[i] * x[i]
      return score
    
    def predict(self, x):
      score = self.score(x)
      return 1 if score >= 0.0 else -1
    
    def get_allw(self):
        return self.allw.copy()
    def get_C(self):
        return self.C.copy()

    def cout(self, desc_set, label_set):
        ret = 0
        i = 0
        while i < len(desc_set):
            tmp = (1 - self.score(desc_set[i])*label_set[i])
            if tmp > 0:
                ret += tmp
            else:
                ret += 0
            i += 1
        return ret

    



    
class ClassifierPerceptronKernel(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, noyau, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.init = init
        res = []
        if init == 0:
          self.w_de_base = np.asarray(np.zeros(self.input_dimension))
        elif init == 1:
          for i in range(self.input_dimension):
            res.append((2*i - 1)*0.001)
          self.w_de_base = np.asarray(res)
        self.noyau = noyau
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """ 
        desc_set = self.noyau.transform(desc_set)

        liste_indice = np.random.permutation(len(desc_set))
        for i in liste_indice:
            x=desc_set[i]
            y=label_set[i]
            pred = self.predict(x)
            if y != pred:
                temp = self.w_de_base + self.learning_rate*x*y
                self.w_de_base = temp

        return self.w_de_base
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """     

        ret = []  
        i = 1
        w_old = self.train_step(desc_set, label_set)
        while i < niter_max:
          w_new = self.train_step(desc_set, label_set)
          d = np.linalg.norm(w_old - w_new)
          ret.append(d)
          #if i != 1:
            #if d < seuil:
              #break
          w_old = w_new.copy()
          i += 1
        ret.pop(0)
        return ret
    
    def score(self,x):
      if len(x) < 6:
          temp = []
          temp.append(float(1.0))
          temp.append(float(x[0]))
          temp.append(float(x[1]))
          temp.append(float(x[0]**2))
          temp.append(float(x[1]**2))
          temp.append(float(x[0]*x[1]))
          x = temp
      score = 0
      for i in range(len(x)):
        score += self.w_de_base[i] * x[i]
      return score
    
    def predict(self, x):
      score = self.score(x)
      return 1 if score >= 0.0 else -1

    
# ---------------------------

# ---------------------------

class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD(desc_set, label_set, self.epsilon, self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        ##################
        ## COMPLETER ICI !
        ##################
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)



class ClassifierArbreNumerique(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision numérique
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD_num(desc_set,label_set,self.epsilon,self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)



