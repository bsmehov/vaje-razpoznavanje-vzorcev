from znacilke import *
from numpy.core.shape_base import block
from sklearn.feature_selection import SelectKBest
import sys, os
from tqdm import tqdm
import pandas as pd


def csm(v1, v2):
    """Kosinusna mera podobnosti med vektorjema v1 in v2."""
    kosinusna_mera_podobnosti = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    # kosinusna_mera_podobnosti = np.sum(v1*v2)/(np.sqrt(np.sum(v1**2))*np.sqrt(np.sum(v2**2)))
    return kosinusna_mera_podobnosti
    # TODO: spiši funkcijo

def euc(v1, v2):
    """Evklidska razdalja med vektorjema v1 in v2."""
    evklidska_razdalja = np.linalg.norm(v1 - v2)
    # evklidska_razdalja = np.sqrt(np.sum((v1-v2)**2))
    return evklidska_razdalja
     # TODO: spiši funkcijo

class CSMKNN():
    """Razpoznavalnik s prileganjem K najbližjih sosedov,
    ki za prileganje lahko uporablja poljubno mero razdalje ali podobnosti."""

    def __init__(self, k=5, mera=csm, nacin="max"):
        """Inicializira razpoznavalnik.
        k: število najbližjih sosedov.
        mera: funkcija, ki kot vhod prejme dva vektorja značilk in vrne
              njuno razdaljo ali podobnost.
        nacin: 'max', če uporabljamo mero podobnosti, oz.
               'min' za mere razdalje."""
        self.k = k
        self.mera = mera
        self.nacin = nacin

        # def _init_ je funkcija za inicializacijo razreda, oziroma objektov v 
        # razredu. V argumentu funkcije podamo self in pa imena spremenljivk, ki 
        # jih želimo zapisati v razred, v našem primeru imajo te spremenljvke že
        # v argumentu zapisane vrednosti

    def fit(self, X, y):
        # oznake grejo od 0 do (N_razredov - 1)
        # "učenje" razpoznavalnika sestoji samo iz koraka kopiranja učne zbirke
        self.N_razredov = y.max() + 1
        self.X = np.copy(X)
        self.y = np.copy(y)

    def predict(self, X):
        """Funkcija, ki izračuna napovedi oznak za vzorce v matriki X."""
        # vektor, kamor bomo shranili izračunane oznake vektorjev X
        y = np.zeros((X.shape[0],), dtype="int32")

        # Izračunamo matriko podobnosti oz. razdalje med vsakim vektorjem 
        # značilk v učni zbirki in vsakim vektorjem značilk v testni zbirki - 
        # self.X je učna zbirka, ki smo jo podali v metodi fit().
        M = np.zeros((X.shape[0], self.X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(self.X.shape[0]):
                M[i, j] = self.mera(X[i], self.X[j])

        # i-ta vrstica matrike M zdaj vsebuje podobnosti med i-tim vektorjem v 
        # testni zbirki ter vsakim izmed vektorjev v učni zbirki.
        # s funkcijo argsort najdemo indekse K najbolj podobnih:

        for i in range(M.shape[0]):
            indeksi = np.argsort(M[i])

            # zanimajo nas najbolj podobni vektorji, ki so za mere podobnosti
            # na koncu razvrščenega seznama, za razdalje pa na začetku:
            if self.nacin == "max":
                sosedi = indeksi[-self.k:]
            elif self.nacin == "min":
                sosedi = indeksi[:self.k]
            else:
                raise ValueError("Nacin mora biti 'min' ali 'max'.")

            # poiščemo oznako vsakega od najbližjih sosedov:
            oznake = [self.y[indeks] for indeks in sosedi]

            # preštejemo, kolikokrat se v vektorju oznak pojavi vsaka izmed
            # oznak razredov:
            pojavi = []
            for j in range(self.N_razredov):
                pojavi_j = np.sum([oznaka == j for oznaka in oznake])
                pojavi.append(pojavi_j)

            # i-temu vektorju v testni zbirki priredimo oznako tistega razreda,
            # ki se med njegovimi K najbližjimi sosedi pojavi največkrat:
            y[i] = np.argmax(pojavi)

        return y

def premesaj_vrstice(X, y):
    """Premeša matriko vektorjev X ter vektor oznak y, z uporabo
    iste naključne permutacije."""
    X_premesan = np.zeros_like(X)
    y_premesan = np.zeros_like(y)

    indeksi = np.random.permutation(X.shape[0])

    for i, indeks in enumerate(indeksi):
        X_premesan[i] = X[indeks]
        y_premesan[i] = y[indeks]

    return X_premesan, y_premesan

def navzkrizno_preverjanje(X, y, N, izbiralnik, razpoznavalnik):
    """N-kratno navzkrižno preverjanje natančnosti razpoznavalnika CSMKNN
    na podatkovni zbirki, podani z vektorji značilk X in vektorjem oznak y."""
    # Najprej naključno premešamo matriko X in vektor y:
    X_premesan, y_premesan = premesaj_vrstice(X, y)

    # podatkovno zbirko enakomerno razdelimo na N kosov:
    X_deli = []
    y_deli = []

    X_deli = np.array_split(X_premesan,N)
    y_deli = np.array_split(y_premesan,N)
    # for i in range(N):
    #     pass
        # TODO: enakomerno razdeli podatkovno bazo na N kosov,
        #       tako, da seznam X_deli vsebuje N matrik značilk oblike
        #       (_, 24), seznam y_deli pa vsebuje N vektorjev oznak
        #       oblike (_,). Vsak del mora vsebovati čim bolj enako porazdelitev
        #       razredov vzorcev.
    
        # Zgled: podatkovno zbirko stotih 10-razsežnih vzorcev iz štirih 
        # različnih razredov želimo razdeliti za 5-kratno navzkrižno preverjanje.
        # V vsaki izmed N=5 iteracij moramo vzeti 20 vzorcev, torej 5 iz vsakega
        # razreda. X_deli bo tedaj seznam petih matrik oblike (20, 10),
        # y_deli pa seznam petih vektorjev oznak oblike (20,).
        # Ob dovolj velikem številu vzorcev lahko približno enakomerno delitev
        # razredov dosežemo z naključnim vzorčenjem.


    
    # Seznam, kamor bomo shranjevali uspešnosti posameznih poskusov
    uspesnosti = []

    # N-krat ponovimo postopek učenja in testiranja, razpoznavalnika pri čemer 
    # v i-tem poskusu testiramo na i-tem delu razdeljene zbirke, 
    # učimo pa na vseh ostalih:
    for i in range(N):
        # matrika učnih vzorcev in njihovih oznak - za učenje uporabimo
        # vseh (N-1) delov podatkovne zbirke, razen i-tega.
        X_train = [X_deli[ind] for ind in range(N) if ind != i]
        y_train = [y_deli[ind] for ind in range(N) if ind != i]

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        # i-ti del podatkovne zbirke bomo uporabili za testiranje naučenega
        # razpoznavalnika.
        X_test = X_deli[i]
        y_test = y_deli[i]

        # Učenje izbiralnika značilk na podlagi učne množice:
        izbiralnik.fit(X_train, y_train)

        # Uporaba naučenega izbiralnika na učnih in testnih značilkah:
        X_train = izbiralnik.transform(X_train)
        X_test  = izbiralnik.transform(X_test)

        # Učenje razpoznavalnika CSMKNN na podlagi učnih značilk:
        razpoznavalnik.fit(X_train, y_train)

        # Napoved oznak razpoznavalnika CSMKNN na testnih značilkah:
        y_hat = razpoznavalnik.predict(X_test)

        # TODO: Izračunaj uspešnost razpoznavalnika na testni zbirki.
        # uspešnost merimo kot delež predvidenih oznak na testni zbirki, ki se
        # ujemajo z dejanskimi:
        uspesnost = 0.0

        y_test1 = np.array(y_test)
        y_hat1 = np.array(y_hat)
        enakosti = y_test1 - y_hat1
        stevilo_enakosti = len(np.where(enakosti==0)[0])
        uspesnost = stevilo_enakosti/len(y_test)
        uspesnosti.append(uspesnost)

    return uspesnosti

def shrani_znacilke():
    dn = "slike/"
    fns = sorted(os.listdir(dn))

    slike = []
    oznake = []

    for fn in fns:
        oznake.append(int(fn[3])-1)

        objekt = cv2.imread("slike/"+fn)
        slike.append(objekt)
        # TODO: Iz datotečnega imena preberi oznako objekta na sliki.
        #       Oznako, ki naj bo 0-4, shrani v seznam oznake.
        # TODO: preberi sliko in jo shrani v seznam slike.


    vektorji = []
    for slika in tqdm(slike):
        vektor = doloci_ffk(slika, 4, 4)
        vektorji.append(vektor)

    X = np.array(vektorji)
    y = np.array(oznake)
    np.save("X.npy", X)
    np.save("y.npy", y)
    return (X, y)

def nalozi_znacilke():
    X = np.load("X.npy")
    y = np.load("y.npy")
    return (X, y)

if __name__ == "__main__":
    # Če datoteki z značilkami in oznakami že obstajata, ju uporabimo.
    # Sicer značilke izračunamo in shranimo.
    if "X.npy" in os.listdir(".") and "y.npy" in os.listdir("."):
        X, y = nalozi_znacilke()
    else:
        X, y = shrani_znacilke()

    # Preverimo uspešnost razpoznavanja za vse kombinacije števila značilk
    # in števila najbližjih sosedov, z evklidsko razdaljo:
    rezultati = []
    for N_znacilk in range(1, 25):
        if N_znacilk == 24:
            N_znacilk = "all"
        izb = SelectKBest(k=N_znacilk)
        rezultati_za_ta_N_znacilk = []
        for k in [1, 3, 5, 7]:
            # print(N_znacilk, k)
            knn = CSMKNN(k, mera=euc, nacin="min")
            uspesnosti = navzkrizno_preverjanje(X, y, 5, izb, knn)
            rezultati_za_ta_N_znacilk.append(np.mean(uspesnosti))
        rezultati.append(rezultati_za_ta_N_znacilk)

    rezultati_kosinusna = []
    for N_znacilk in range(1, 25):
        if N_znacilk == 24:
            N_znacilk = "all"
        izb = SelectKBest(k=N_znacilk)
        rezultati_za_ta_N_znacilk = []
        for k in [1, 3, 5, 7]:
            # print(N_znacilk, k)
            knn = CSMKNN(k, mera=csm, nacin="max")
            uspesnosti = navzkrizno_preverjanje(X, y, 5, izb, knn)
            rezultati_za_ta_N_znacilk.append(np.mean(uspesnosti))
        rezultati_kosinusna.append(rezultati_za_ta_N_znacilk)

    # TODO: Preverite  še uspešnost razpoznavanja pri vseh kombinacijah z mero
    #       kosinusne podobnosti. Rezultate podajte v tabelah.

    plt.figure()
    plt.title("Evklidska razdalja")
    plt.xlabel('k')
    plt.ylabel('N. znacilk')
    plt.imshow(rezultati)
    plt.xticks(np.arange(0,4), np.arange(1, 8, step=2))
    plt.yticks(np.arange(len(rezultati)),np.arange(1, len(rezultati)+1))
    plt.colorbar()
    plt.tight_layout()
    plt.show(block=False)

    plt.figure()
    plt.title("Kosinusna podobnost")
    plt.xlabel('k')
    plt.ylabel('N. znacilk')
    plt.imshow(rezultati_kosinusna)
    plt.xticks(np.arange(0,4), np.arange(1, 8, step=2))
    plt.yticks(np.arange(len(rezultati_kosinusna)),np.arange(1, len(rezultati_kosinusna)+1))
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # TODO: Preverite  še uspešnost razpoznavanja pri vseh kombinacijah z mero
    #       kosinusne podobnosti. Rezultate podajte v tabelah.
    n_znac=[]
    n_znac.extend(range(1,25))
    data1 = pd.DataFrame(data = rezultati, index=n_znac, columns=['1', '3','5','7'])
    data1.to_excel('euc.xlsx',index_label='znacilke/K',sheet_name='euc')


    data2=pd.DataFrame(data = rezultati_kosinusna, index=n_znac, columns=['1', '3','5','7'])
    data2.to_excel('cos.xlsx',index_label='znacilke/K',sheet_name='cos')