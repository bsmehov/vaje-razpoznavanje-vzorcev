from obrisi import *

def obdelaj_sliko(slika):
    """Pretvorba v sivinsko sliko, filtriranje, upragovljanje."""
    sivaSlika = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
    sivaSlika = cv2.resize(sivaSlika, dsize=(256, 256),
                           interpolation=cv2.INTER_CUBIC)
    obdelanaSlika = cv2.medianBlur(sivaSlika, 11)
    prag = dolociPrag(obdelanaSlika)
    _, upragovljenaSlika = cv2.threshold(obdelanaSlika, prag, 255, 0)
    return upragovljenaSlika

def pretvori_obris_v_signal(obris):
    """Pretvorba obrisa,
    iz zapisa v obliki zaporedja točk v kompleksni signal."""
    tocke_obrisa = obris["tocke"]
    dolzina = len(tocke_obrisa)
    signal = np.zeros((dolzina,), dtype=np.complex128)
    for i in range(dolzina):
        x, y = tocke_obrisa[i]
        signal[i] = x + 1j * y
    return signal

def prevzorci_signal(signal, N_nov=64):
    """Prevzorčenje signala poljubne dolžine na fiksno število točk."""
    N_orig = signal.shape[0]
    nov_signal = np.zeros((N_nov,), dtype=signal.dtype)
    for i in range(N_nov):
        i_r = i / (N_nov - 1)
        j_r = i_r * (N_orig - 1)

        if j_r == 0 or j_r == (N_orig - 1):
            nov_signal[i] = signal[int(j_r)]
        else:
            j0 = int(j_r)
            j1 = j0 + 1
            t = j_r - j0

            y0 = signal[j0]
            y1 = signal[j1]
            nov_signal[i] = y0 * (1 - t) + y1 * t

    return nov_signal

def doloci_ffk(slika, kmax, lmax):
    """Določitev vektorja značilk iz najdaljšega obrisa na sliki."""
    binarnaSlika = obdelaj_sliko(slika)

    iskalnikObrisov = Iskalnik(binarnaSlika)
    iskalnikObrisov.isci_obrise()
    obris = iskalnikObrisov.podaj_najdaljsi_obris()

    signal = pretvori_obris_v_signal(obris)
    signal = prevzorci_signal(signal)

    signal_fft = np.fft.fft(signal)

    vektor_ffk = np.zeros((2 * kmax * (lmax - 1),), dtype=np.float64)
    ind_vektor = 0

    for i in range(1, kmax + 1):
        for j in range(2, lmax + 1):
            F_k = signal_fft[i + 1] ** j
            F_l = signal[-j + 1] ** i
            F1 = signal_fft[1] ** (i + j)
            d_ij = F_k * F_l / F1
            d0 = d_ij.real
            d1 = d_ij.imag
            vektor_ffk[ind_vektor] = d0
            vektor_ffk[ind_vektor + 1] = d1
            ind_vektor += 2

    return vektor_ffk
