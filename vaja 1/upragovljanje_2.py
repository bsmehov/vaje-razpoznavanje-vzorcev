import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

def izracunajHistogram(sivaSlika):
    """Izračun histograma sivinske slike.
    @param sivaSlika : uint8 2D numpy array poljubne velikosti
    izhod: histogram slike"""
    histogram = np.zeros((256,), dtype="float64")
    for sivi_nivo in range(256):
        st_pikslov = np.sum(sivaSlika == sivi_nivo)
        histogram[sivi_nivo] = st_pikslov
    return histogram

def narisiHistogram(histogram):
    """Izris izračunanega histograma.
    @param histogram : 1D numpy array poljubne velikosti in tipa"""
    n_elementov = histogram.shape[0]
    plt.figure()
    plt.title("histogram")
    plt.bar(np.arange(n_elementov), histogram)
    plt.xlabel("svetilnost")
    plt.ylabel("st. pikslov")
    plt.grid(True)
    return 0

def dolociPrag(sivaSlika):
    """Določitev praga sive slike po metodi maksimizacjie informacije.
    @param sivaSlika : slika, ki ji določamo prag
    izhod: vrednost praga, ki maksimizira informacijo"""
    histogram = izracunajHistogram(sivaSlika)

    # izračun števila pikslov v sliki
    n = sivaSlika.shape[0] * sivaSlika.shape[1]

    # izračun porazdelitve relativnih frekvenc, P
    P = histogram / n

    # inicializacija vektorja, kamor bomo shranjevali informacijo
    # za vsako možno vrednost praga
    informacija = np.zeros_like(histogram)

    imenovalec = 0
    H0 = np.zeros_like(histogram)
    H1 = np.zeros_like(histogram)

    for t in range(1,255):
        
        imenovalec += P[t]

        for i in range(1, 255):
            if (i <= t and imenovalec>0 and P[i]>0):
                H0[t] -= ((P[i]/imenovalec)*np.log(P[i]/imenovalec))
            
            if (i > t and imenovalec<1 and P[i]>0):
                H1[t] -= ((P[i]/(1-imenovalec))*np.log(P[i]/(1-imenovalec)))

        informacija[t]= H0[t]+H1[t]

    
    #TODO: izračunaj informacijo pri vsaki možni vrednosti praga
    #TODO: določi vrednost praga, ki maksimizira informacijo

    # izračun informacije pri vsakem možnem pragu:
    # iskanje vrednosti praga, kjer je informacija maksimalna

    t = np.argmax(informacija)

    return t

if __name__ == "__main__":
    # Preberi sliko s podanega datotečnega imena
    try:
        filename = sys.argv[1]
    except IndexError:
        print("Uporaba programa: python upragovljanje.py <datotečno ime slike>")
        sys.exit(1)

    # Branje slike in pretvorba barvnega prostora
    slika = cv2.imread(filename)
    slika = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
    sivaSlika = cv2.cvtColor(slika, cv2.COLOR_RGB2GRAY)

    # TODO: Obdelava sivinske slike, npr. s postopki
    # izravnave histograma, filtrom mediane in Gaussovim glajenjem
    # gl. funkcije: cv2.equalizeHist, cv2.GaussianBlur, cv2.medianBlur

    obdelanaSlika = cv2.medianBlur(sivaSlika,9)

    # Izračun in prikaz histograma ter slik

    histogram = izracunajHistogram(obdelanaSlika)
    narisiHistogram(histogram)
    prag = dolociPrag(obdelanaSlika)

    _, upragovljenaSlika = cv2.threshold(obdelanaSlika, prag, 255, cv2.THRESH_BINARY)
    
    print(prag)

    plt.figure()
    plt.title("Barvna slika")
    plt.imshow(slika)

    plt.figure()
    plt.title("Sivinska slika")
    plt.imshow(sivaSlika, cmap="gray")

    plt.figure()
    plt.title("Obdelana slika")
    plt.imshow(obdelanaSlika, cmap="gray")

    plt.figure()
    plt.title("upragovljenaSlika")
    plt.imshow(upragovljenaSlika, cmap="gray")

    # TODO: dopolnite funkcijo za izračun praga, dolociPrag
    # TODO: uporabite funkcijo dolociPrag za določitev praga obdelane sive slike
    # TODO: uporabite izračunano vrednost praga za upragovljanje sivinske slike
    # TODO: izpišite izračunano vrednost praga ter prikažite upragovljeno sliko,
    #       POLEG barvne in sivinske slike.

    #prag = dolociPrag(obdelanaSlika)
    
    #_, upragovljenaSlika = cv2.threshold(obdelanaSlika, prag, 255, 0)
    
    plt.show()

    input("Press any key.")
    plt.close("all")
    sys.exit(0)