import numpy as np
import matplotlib.pyplot as plt
import cv2

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

    #TODO: izračunaj informacijo pri vsaki možni vrednosti praga
    #TODO: določi vrednost praga, ki maksimizira informacijo

    # izračun informacije pri vsakem možnem pragu:
    for t in range(1, 255):
        Pstar = np.sum(P[:t])

        H0 = 0
        for i in range(0, t):
            if P[i] > 0 and Pstar > 0:
                term = P[i] / Pstar
                H0 -= (term * np.log(term))
        
        H1 = 0
        for i in range(t, 255):
            if P[i] > 0 and (1 - Pstar) > 0:
                term = P[i] / (1 - Pstar)
                H1 -= (term * np.log(term))

        informacija[t] = H0 + H1

    # iskanje vrednosti praga, kjer je informacija maksimalna
    t = np.argmax(informacija)

    print(t);

    return t
