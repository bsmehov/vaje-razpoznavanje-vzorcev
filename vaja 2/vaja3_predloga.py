import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

def histogram_fonemov(y, fonemi):
    histogram = [(y==i).sum() for i in range(len(fonemi))]
    plt.bar(np.arange(len(fonemi)), histogram)
    plt.xticks(np.arange(len(fonemi)), fonemi)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def pripravi_zbirko(filename):
    """Prebere podatkovno zbirko iz datoteke .arff ter jo vrne v obliki,
    primerni za učenje in preizkušanje sklearn modelov.
    vhod: datotečno ime .arff datoteke
    izhodi: X - matrika značilk vzorcev
            y - matrika številčnih oznak vzorcev
            fonemi - seznam fonemov, ki dekodira številčne oznake"""
    with open(filename, "r") as f:
        vsebina = f.read()
    vrstice = vsebina.split("\n")[:-1]
    

    fonemi = "@,E,O,S,W,a,b,d,e,f,g,i,j,k,l,m,n,o,p,r,s,sil,t,tlesk,ts,u,v,vdih,w,x,z".split(",")

    y_seznam = []
    seznam = []
    X_seznam = []
    zacetek = vrstice.index("@DATA")
    konec = vrstice.index(vrstice[-1])
    for i in range(zacetek+1, konec+1):
        seznam = vrstice[i].split(",")
        y_seznam.append(seznam[-1])
        X_seznam.append(seznam[0:-1])


    # Naredimo vektor y
    y_nov = []
    for element in y_seznam:
        y_nov.append(fonemi.index(element))


    # Shranimo seznam X in y v matriko oz. vektor
    X = np.array(X_seznam)
    y = np.array(y_nov)


    # prikaz porazdelitve fonemov
    histogram_fonemov(y, fonemi)

    # TODO: preberi značilke vzorcev in jih shrani v matriko X
    # TODO: preberi foneme in jih shrani v seznam fonemi
    # TODO: preberi oznake vzorcev, jih kodiraj tako, da indeksirajo
    #       seznam fonemi, in shrani v vektor y

    return X, y, fonemi

def premesaj_vrstice(X, y):
    inds = np.random.permutation(X.shape[0])
    return X[inds], y[inds]

def konfuzijska_matrika(y_test, y_hat, fonemi):
    """Prikaže konfuzijsko matriko razpoznavalnika na podlagi
    pravilnih oznak (y_test), izračunanih oznak (y_hat) ter seznama fonemov."""
    matrika = np.zeros((len(fonemi), len(fonemi)), dtype="int32")

    for i in range(len(y_test)):
        matrika[y_test[i],y_hat[i]] += 1

    vsota_matrike = np.sum(matrika)
    nova_matrika = matrika/vsota_matrike

    diagonala_fonemov = np.diagonal(nova_matrika)

    najbolj_prepoznan_f = np.argmax(diagonala_fonemov)
    najmanj_prepoznan_f = np.argmin(diagonala_fonemov)

    print("najbolje prepoznan:", fonemi[najbolj_prepoznan_f])
    print("najmanj prepoznan:", fonemi[najmanj_prepoznan_f])

    for i, element in enumerate(nova_matrika):
        indeks = np.argsort(element)[::-1]
        print("za fonem: ", fonemi[i])
        if i != indeks[0]:
            print("najpogostejša napaka: ", fonemi[indeks[0]])
        else:
            print("najpogostejša napaka: ", fonemi[indeks[1]])


    #TODO: sestavi konfuzijsko matriko
    
    plt.imshow(matrika)
    plt.xticks(np.arange(len(fonemi)), fonemi)
    plt.yticks(np.arange(len(fonemi)), fonemi)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def navzkrizno_preverjanje(X, y, N, razvrscevalnik, fonemi):
    """Navzkrižno preverjanje natančnosti razpoznavalnika
    na podatkovni zbirki, podani z vektorji značilk X in vektorjem oznak y."""
    # Najprej naključno premešamo matriko X in vektor y:
    X_premesan, y_premesan = premesaj_vrstice(X, y)

    # podatkovno zbirko enakomerno razdelimo na N kosov:
    X_deli = []
    y_deli = []
    for i in range(N):
        i0 = (X.shape[0] // N) * i
        i1 = (X.shape[0] // N) * (i + 1)

        X_del = X_premesan[i0 : i1]
        y_del = y_premesan[i0 : i1]

        X_deli.append(X_del)
        y_deli.append(y_del)

    # Seznam, kamor bomo shranjevali uspešnosti poskusov
    uspesnosti = []

    # N-krat ponovimo postopek učenja in testiranja, pri čemer v i-tem poskusu
    # testiramo na i-tem delu razdeljene zbirke, učimo pa na vseh ostalih:
    for i in range(N):
        X_train = [X_deli[ind] for ind in range(N) if ind != i]
        y_train = [y_deli[ind] for ind in range(N) if ind != i]

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        X_test = X_deli[i]
        y_test = y_deli[i]

        razvrscevalnik.fit(X_train, y_train)

        y_hat = razvrscevalnik.predict(X_test)

        # uspešnost merimo kot delež predvidenih oznak na testni zbirki, ki s
        # ujemajo z dejanskimi:
        uspesnost = np.mean(y_hat == y_test)
        uspesnosti.append(uspesnost)

        if i == 0:
            konfuzijska_matrika(y_test, y_hat, fonemi)

    return uspesnosti

if __name__ == "__main__":
    
    X, y, fonemi = pripravi_zbirko("govorec1.arff")

    seznam_k = [1,3,5,7,9]

    for k in seznam_k:
        najblizji_sosed = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        uspesnosti = navzkrizno_preverjanje(X, y, 5, najblizji_sosed, fonemi)
        povprecna_uspesnost = np.mean(uspesnosti)
        print("uspesnost KNN: ", uspesnosti)
        print("povprecna uspesnost: ", povprecna_uspesnost)
    
    for c in [1, 10, 100, 1000]:
        svm1 = SVC(C = c,kernel = 'rbf', degree = 3, gamma = 'scale')
        uspesnosti1 = navzkrizno_preverjanje(X, y, 5, svm1, fonemi)
        povprecna_uspesnost1 = np.mean(uspesnosti1)
        print("uspesnost SVM s C-ji: ", uspesnosti1)
        print("povprecna uspesnost s C-ji, rbf, degree 3: ", povprecna_uspesnost1)

    for c in [1, 10, 100, 1000]:
        svm2 = SVC(C = c,kernel = 'poly', degree = 2, gamma = 'scale')
        uspesnosti2 = navzkrizno_preverjanje(X, y, 5, svm2, fonemi)
        povprecna_uspesnost2 = np.mean(uspesnosti2)
        print("uspesnost SVM s C-ji: ", uspesnosti2)
        print("povprecna uspesnost s C-ji, poly, degree 2: ", povprecna_uspesnost2)

    
    


    # TODO: preizkusi razpoznavanje na dobljeni podatkovni zbirki z obema 
    #       razpoznavalnikoma. Za vsak razpoznavalnik najdi najbolj ustrezne
    #       vrednosti parametrov. Izračunaj povprečno uspešnost razpoznavanja
    #       z uporabo 5-kratnega navzkrižnega preverjanja. Za prvo izmed
    #       iteracij prikaži konfuzijsko matriko.