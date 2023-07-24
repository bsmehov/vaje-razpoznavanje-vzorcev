import sys, os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from numpy import random
import pandas as pd
import itertools
import math
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial import distance

def euc(tocka1, tocka2):
    euc_distance = distance.euclidean(tocka1,tocka2)
    return euc_distance

def pretvorba(seznam_string):
    seznam_pretvorb = []
    for i in range(len(seznam_string)-1):
        tocka = seznam_string[i].split(",")
        tocka[0] = float(tocka[0])
        tocka[1] = float(tocka[1])
        tocka[2] = float(tocka[2])
        tocka[3] = float(tocka[3])
        seznam_pretvorb.append(tocka[:4])
        if i == 1:
            print(seznam_pretvorb)

    return seznam_pretvorb

def kofeneticna_matrika(seznam_tock1,tocke,euc_za_rac):

    l = len(seznam_tock1)
    dict={}
    for i in range(len(tocke)):
        dict[i+l]=list(tocke[i])

    nova = np.zeros((l,l))

    for k in dict:
        h = dict[k]
        e = h.copy()
        if e[0]>l-1:
            e[0] = dict[e[0]]
        else:
            e[0]=[e[0]]
        if e[1]>l-1:
            e[1] = dict[e[1]]
        else:
            e[1]=[e[1]]
        c = list(itertools.product(e[0], e[1]))
        for x in c:
            x=list(x)
            if(x[1] > x[0]):
                x_1 = x[0]
                x[0] = x[1]
                x[1] = x_1
            nova[x[0]][x[1]] = euc_za_rac[h[0]][h[1]]
            
        enovi=[]
        enovi.extend(e[0])
        enovi.extend(e[1])
        dict[k] = enovi

    return nova

def racunanje_rojev(evklidove_razdalje,euc_za_rac,tocke1,stevec,N,seznam_razdalj,tocke,a,stevilo_vzorcev):

    if(a == 1):
        a_i = 0.5
        a_j = 0.5
        b = 0
        c = -0.5
    if(a == 2):
        a_i = 0.5
        a_j = 0.5
        b = 0
        c = 0.5

    while(N > 1):
                
        seznam_razdalj1 = []

        for i in range(len(evklidove_razdalje)):         
            if(a == 3):
                imenovalec = (stevilo_vzorcev[i] + stevilo_vzorcev[tocke1[stevec]] + stevilo_vzorcev[tocke1[stevec+1]])
                a_i = (stevilo_vzorcev[i] + stevilo_vzorcev[tocke1[stevec+1]])/imenovalec
                a_j = (stevilo_vzorcev[i] + stevilo_vzorcev[tocke1[stevec]])/imenovalec
                b = -stevilo_vzorcev[i]/imenovalec
                c = 0

            razdalja = a_i*euc_za_rac[tocke1[stevec]][i] + a_j*euc_za_rac[tocke1[stevec+1]][i] + b*euc_za_rac[tocke1[stevec]][tocke1[stevec+1]] + c*abs(euc_za_rac[tocke1[stevec]][i] - euc_za_rac[tocke1[stevec+1]][i])
            seznam_razdalj1.append(razdalja)

        seznam_razdalj1_np = np.array([seznam_razdalj1])
        
        evklidove_razdalje = np.append(evklidove_razdalje, seznam_razdalj1_np, 0)
        euc_za_rac = np.append(euc_za_rac, seznam_razdalj1_np, 0)

        seznam_razdalj1.append(15)

        seznam_razdalj2_np = np.array([seznam_razdalj1])

        seznam_razdalj1_rotiran = np.rot90(seznam_razdalj2_np, axes = (1,0))

        evklidove_razdalje = np.append(evklidove_razdalje, seznam_razdalj1_rotiran, 1)
        euc_za_rac = np.append(euc_za_rac, seznam_razdalj1_rotiran, 1)

        for i in tocke1:
            for j in range(len(evklidove_razdalje)):
                evklidove_razdalje[i][j] = 15
                evklidove_razdalje[j][i] = 15

        stevilo_vzorcev.append(stevilo_vzorcev[tocke1[stevec]]+stevilo_vzorcev[tocke1[stevec+1]])
        stevec = stevec + 2

        tocke1 = []

        minimum = np.amin(evklidove_razdalje)
        index = np.where(evklidove_razdalje == minimum)

        N = N - 1

        seznam_razdalj.append(minimum)

        lista_tock = list(zip(index[0], index[1]))

        for i in range(1):
            cord = lista_tock[i]
            tocke.append(cord)

        for i in range(len(tocke)):
            tocka = tocke[i]
            tocke1.append(tocka[0])
            tocke1.append(tocka[1])

    return tocke, seznam_razdalj, euc_za_rac

def izracun_Q(kofeneticna_matrika_1,evklidove_razdalje_za_Q):
    stevec = 0
    imenovalec = 0
    for i in range(len(kofeneticna_matrika_1)-1):
        for j in range(i+1,len(kofeneticna_matrika_1)):
            stevec = stevec + (evklidove_razdalje_za_Q[j][i] - kofeneticna_matrika_1[j][i])**2
            imenovalec = imenovalec + (evklidove_razdalje_za_Q[j][i])**2
    Q = stevec/imenovalec
    return Q

def izracun_CPCC(kofeneticna_matrika_1,evklidove_razdalje_za_Q):
    N = len(kofeneticna_matrika_1)
    R = N*(N - 1)/2
    stevec1 = 0
    D = 0
    D_c = 0
    imenovalec1 = 0
    imenovalec2 = 0

    for i in range(N - 1):
        for j in range(i+1,N):
            stevec1 = stevec1 + evklidove_razdalje_za_Q[j][i]*kofeneticna_matrika_1[j][i]
            D = D + evklidove_razdalje_za_Q[j][i]
            D_c = D_c + kofeneticna_matrika_1[j][i]
            imenovalec1 = imenovalec1 + evklidove_razdalje_za_Q[j][i]**2
            imenovalec2 = imenovalec2 + kofeneticna_matrika_1[j][i]**2

    crta_D = (1/R)*D
    crta_D_c = (1/R)*D_c
    stevec1 = (1/R)*stevec1
    stevec2 = crta_D*crta_D_c
    stevec = stevec1 - stevec2
    
    imenovalec1 = math.sqrt((1/R)*imenovalec1-crta_D**2)
    imenovalec2 = math.sqrt((1/R)*imenovalec2-crta_D_c**2)
    imenovalec = imenovalec1*imenovalec2

    CPCC = stevec/imenovalec
    return CPCC
            

if __name__ == "__main__":

    with open('D:/ljubljana/razpoznavanje_vzorcev_real/izbirni_projekt/irisdata.txt') as f:
        tocke = f.readlines()

    seznam_tock1 = pretvorba(tocke)
    seznam_tock = np.array(seznam_tock1)

    N = len(seznam_tock)

    print("dolzina: ")
    print(N)

    evklidove_razdalje = np.zeros((len(seznam_tock),len(seznam_tock)))
    seznam_zdruzitev = []

    for i in range(len(seznam_tock)):
        for j in range(len(seznam_tock)):
            evklidove_razdalje[i][j] = euc(seznam_tock[i],seznam_tock[j])
    
    
    evklidove_razdalje = np.where(evklidove_razdalje == 0, 15 , evklidove_razdalje)
    evklidove_razdalje_za_Q = np.copy(evklidove_razdalje)
    
    euc_za_rac = np.zeros((len(seznam_tock),len(seznam_tock)))

    for i in range(len(seznam_tock)):
        for j in range(len(seznam_tock)):
            euc_za_rac[i][j] = euc(seznam_tock[i],seznam_tock[j])
    
    euc_za_rac = np.where(euc_za_rac == 0, 15 , euc_za_rac)

    seznam_razdalj = []
    minimum = np.amin(evklidove_razdalje)
    index = np.where(evklidove_razdalje == minimum)

    N = N - 1

    seznam_razdalj.append(minimum)
    seznam_razdalj_2 = seznam_razdalj.copy()
    seznam_razdalj_3 = seznam_razdalj.copy()

    lista_tock = list(zip(index[0], index[1]))
    tocke = []

    for cord in lista_tock:
        if(cord in tocke or cord[::-1] in tocke):
            continue
        tocke.append(cord)

    tocke1 = []

    for i in range(len(tocke)):
        tocka1 =  tocke[i]
        tocke1.append(tocka1[0])
        tocke1.append(tocka1[1])
    
    tocke1_1 = np.copy(tocke1)
    tocke1_2 = np.copy(tocke1)

    tocke2 = tocke.copy()
    tocke3 = tocke.copy()

    for i in tocke1:
        for j in range(len(seznam_tock)):
            evklidove_razdalje[i][j] = 15
            evklidove_razdalje[j][i] = 15
    
    evklidove_razdalje1 = np.copy(evklidove_razdalje)
    euc_za_rac1 = np.copy(euc_za_rac)
    evklidove_razdalje2 = np.copy(evklidove_razdalje)
    euc_za_rac2 = np.copy(euc_za_rac)

    stevec = 0
    N1 = len(seznam_tock) - 1
    stevec1 = 0
    N3 = len(seznam_tock) - 1

    print("N: ")
    print(N1)
    print(N3)
    print(N)
    stevec2 = 0
    stevilo_vzorcev1 = []
    for i in range(len(seznam_tock1)):
        stevilo_vzorcev1.append(1)

    stevilo_vzorcev2 = stevilo_vzorcev1.copy()
    stevilo_vzorcev3 = stevilo_vzorcev1.copy()

    tocke, seznam_razdalj, euc_za_rac = racunanje_rojev(evklidove_razdalje, euc_za_rac, tocke1, stevec, N, seznam_razdalj, tocke, 1, stevilo_vzorcev1)
    tocke2, seznam_razdalj_2, euc_za_rac1 = racunanje_rojev(evklidove_razdalje1, euc_za_rac1, tocke1_1, stevec1, N1, seznam_razdalj_2, tocke2, 2, stevilo_vzorcev2)
    tocke3, seznam_razdalj_3, euc_za_rac2 = racunanje_rojev(evklidove_razdalje2, euc_za_rac2, tocke1_2, stevec2, N3, seznam_razdalj_3, tocke3, 3, stevilo_vzorcev3)

    # vsota = sum(euc_za_rac - euc_za_rac4)
    # print(vsota)
    print("rezultat 1:")
    print(tocke)
    print(seznam_razdalj)
    print("rezultat 2:")
    print(tocke2)
    print(seznam_razdalj_2)
    print("rezultat 3:")
    print(tocke3)
    print(seznam_razdalj_3)

    kofeneticna_matrika_1 = kofeneticna_matrika(seznam_tock1,tocke,euc_za_rac)
    kofeneticna_matrika_2 = kofeneticna_matrika(seznam_tock1,tocke2,euc_za_rac1)
    kofeneticna_matrika_3 = kofeneticna_matrika(seznam_tock1,tocke3,euc_za_rac2)
    df = pd.DataFrame(kofeneticna_matrika_1)
    df.to_excel(excel_writer = "C:/Users/bsmeh/Documents/dodatna_naloga_kofeneticna_matrika_1.xlsx")
    df = pd.DataFrame(kofeneticna_matrika_2)
    df.to_excel(excel_writer = "C:/Users/bsmeh/Documents/dodatna_naloga_kofeneticna_matrika_2.xlsx")
    df = pd.DataFrame(kofeneticna_matrika_3)
    df.to_excel(excel_writer = "C:/Users/bsmeh/Documents/dodatna_naloga_kofeneticna_matrika_3.xlsx")

    Q1 = izracun_Q(kofeneticna_matrika_1,evklidove_razdalje_za_Q)
    Q2 = izracun_Q(kofeneticna_matrika_2,evklidove_razdalje_za_Q)
    Q3 = izracun_Q(kofeneticna_matrika_3,evklidove_razdalje_za_Q)
    CPCC1 = izracun_CPCC(kofeneticna_matrika_1,evklidove_razdalje_za_Q)
    CPCC2 = izracun_CPCC(kofeneticna_matrika_2,evklidove_razdalje_za_Q)
    CPCC3 = izracun_CPCC(kofeneticna_matrika_3,evklidove_razdalje_za_Q)
    print("Q1: ")
    print(Q1)
    print("CPCC1: ")
    print(CPCC1)
    print("Q2: ")
    print(Q2)
    print("CPCC2: ")
    print(CPCC2)
    print("Q3: ")
    print(Q3)
    print("CPCC3: ")
    print(CPCC3)

    maksimum = 0

    for i in range(len(seznam_razdalj_3)-1):
        razdalja = seznam_razdalj_3[i+1] - seznam_razdalj_3[i]
        if razdalja > maksimum:
            maksimum = razdalja
            spodnja_meja_reza = seznam_razdalj_3[i]
            index = i

    indeksi_roj_za_rac = []
    seznam_razdalj_za_rac = seznam_razdalj_3[index:]

    print("seznam_razdalj: ")
    print(seznam_razdalj[index:])

    for i in range(len(seznam_razdalj_za_rac)):
        vrednost = seznam_razdalj_za_rac[i]
        if(vrednost == 15):
            vrednost = seznam_razdalj_2[index-1]
        indeks_roj = np.where(kofeneticna_matrika_3 == vrednost)
        lista_tock = list(zip(indeks_roj[0],indeks_roj[1]))
        tocke_roj = []
        for cord in lista_tock:
            if(all(cord[0] not in k for k in indeksi_roj_za_rac)):
                if(cord[0] not in tocke_roj):
                    tocke_roj.append(cord[0])
            if(all(cord[1] not in k for k in indeksi_roj_za_rac)):
                if(cord[1] not in tocke_roj):
                    tocke_roj.append(cord[1])
        print("vrednost: ")
        print(vrednost)
        indeksi_roj_za_rac.append(tocke_roj)
        print(len(indeksi_roj_za_rac[i]))

    X = []

    for i in range(len(evklidove_razdalje_za_Q)-1):
        for j in range(i+1,len(evklidove_razdalje_za_Q)):
            X.append(evklidove_razdalje_za_Q[j][i])

    Z = linkage(X, 'complete')
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z)
    plt.show()

    # kofeneticna = cophenet(Z)

    # Y = []

    # for i in range(len(kofeneticna_matrika_1)-1):
    #     for j in range(i+1,len(kofeneticna_matrika_1)):
    #         Y.append(kofeneticna_matrika_1[j][i])

    # vsota = sum(Y - kofeneticna)

    # print(vsota)

    # df = pd.DataFrame(kofeneticna_matrika_1)
    # df.to_excel(excel_writer = "C:/Users/bsmeh/Documents/test7.xlsx")


    