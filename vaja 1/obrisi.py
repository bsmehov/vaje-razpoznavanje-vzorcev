from upragovljanje_2 import *
from enum import Enum

class Smer(Enum):
    DESNO = 0
    DOL = 1
    LEVO = 2
    GOR = 3
    NIKAMOR = 4

class Iskalnik():
    def __init__(self, slika):
        """Inicializira iskalnik obrisov v dani binarni sliki."""
        self.slika = slika
        self.obrisi = []

        # dimenzije slike
        self.m = slika.shape[0]
        self.n = slika.shape[1]

        # inicializacija seznama preverjenih elementov
        self.SPE = []

        # inicializacija pomožnih oznak
        self.oznake = np.zeros((self.m, self.n), dtype=np.unicode_)
        for i in range(self.m):
            for j in range(self.m):
                self.oznake[i, j] = "N"

    def isci_obrise(self):
        """Poišče vse obrise na sliki."""
        m, n = self.slika.shape
        self.obrisi = []
        for y in range(m):
            self.SPE = []
            for x in range(n):
                if self.je_zacetna_tocka(x, y):
                    self.sledenje_obrisu(x, y)

        # pri binarnih slikah je prvi obris tipično okvir celotne slike,
        # zato ga izbrišemo.
        if len(self.obrisi) > 1:
            self.obrisi = self.obrisi[1:]

    def je_zacetna_tocka(self, x, y):
        """Preveri, če je dana točka (x, y) veljavna začetna točka obrisa."""
        if self.oznake[y, x] == "N" and \
           (len(self.SPE) == 0 or self.SPE[-1] != self.slika[y, x]):
            return True
        elif self.oznake[y, x] == "A":
            self.SPE.append(self.slika[y, x])
        elif (self.oznake[y, x] == "D" and \
              len(self.SPE) != 0 and \
              self.SPE[-1] == self.slika[y, x]):
            self.SPE.pop()
        return False

    def preveri_okolico(self, x_z, y_z):
        """če je začetna točka obrisa obkoljena s piksli različne barve,
        ni veljavna."""
        oznaka = self.slika[y_z, x_z]
        if self.slika[y_z, x_z + 1] != oznaka and \
           self.slika[y_z, x_z - 1] != oznaka and \
           self.slika[y_z + 1, x_z] != oznaka and \
           self.slika[y_z - 1, x_z] != oznaka:
               return False
        return True

    def sledenje_obrisu(self, x_z, y_z):
        """Sledi obrisu z začetno točko (x_z, y_z)."""
        x_o = x_z
        y_o = y_z

        check = self.preveri_okolico(x_z, y_z)
        if check == False:
            return 0

        koda = []
        tocke = [(x_z, y_z)]

        vstop = Smer.NIKAMOR

        while True:
            izstop = self.premik(x_o, y_o, vstop)
            self.doloci_oznako(vstop, izstop, x_o, y_o)
            if izstop == Smer.GOR:
                y_o -= 1
            elif izstop == Smer.DOL:
                y_o += 1
            elif izstop == Smer.LEVO:
                x_o -= 1
            elif izstop == Smer.DESNO:
                x_o += 1

            koda.append(izstop.value)
            tocke.append((x_o, y_o))
            vstop = izstop
            if x_o == x_z and y_o == y_z:
                break

        zacetna_tocka = (x_z, y_z)
        obris = {"ZT": zacetna_tocka, "kode": koda, "tocke": tocke}
        self.obrisi.append(obris)
        return 0

    def pogled(self, x_o, y_o, smer):
        """Preveri, če je piksel v smeri 'smer' del istega obrisa,
        kot točka (x_o, y_o), z upoštevanjem robnih pogojev."""
        if smer == Smer.DESNO:
            if x_o + 1 >= self.slika.shape[1]:
                return False
            elif self.slika[y_o, x_o + 1] != self.slika[y_o, x_o]:
                return False
            else:
                return True
        elif smer == Smer.DOL:
            if y_o + 1 >= self.slika.shape[0]:
                return False
            elif self.slika[y_o + 1, x_o] != self.slika[y_o, x_o]:
                return False
            else:
                return True
        elif smer == Smer.LEVO:
            if x_o == 0:
                return False
            elif self.slika[y_o, x_o - 1] != self.slika[y_o, x_o]:
                return False
            else:
                return True
        elif smer == Smer.GOR:
            if y_o == 0:
                return False
            elif self.slika[y_o - 1, x_o] != self.slika[y_o, x_o]:
                return False
            else:
                return True
    
    def premik(self, x_o, y_o, vstop):
        """Izvede korak premika po obrisu iz točke (x_o, y_o) v naslednjo
        točko, po pravilu leve prioritete."""
        if vstop == Smer.NIKAMOR or vstop == Smer.DESNO:
            prioriteta = [Smer.GOR, Smer.DESNO, Smer.DOL, Smer.LEVO]
        elif vstop == Smer.DOL:
            prioriteta = [Smer.DESNO, Smer.DOL, Smer.LEVO, Smer.GOR]
        elif vstop == Smer.LEVO:
            prioriteta = [Smer.DOL, Smer.LEVO, Smer.GOR, Smer.DESNO]
        elif vstop == Smer.GOR:
            prioriteta = [Smer.LEVO, Smer.GOR, Smer.DESNO, Smer.DOL]

        for smer in prioriteta:
            if self.pogled(x_o, y_o, smer):
                return smer

    def doloci_oznako(self, vstop, izstop, x_o, y_o):
        """Določi pomožno oznako točke obrisa (x_o, y_o), glede na njeno
        prejšnjo oznako ter smer vstopa in izstopa iz točke."""

        prejsnja = self.oznake[y_o, x_o]

        if vstop == Smer.GOR or vstop == Smer.LEVO or vstop == Smer.NIKAMOR:
            if izstop == Smer.GOR or izstop == Smer.DESNO:
                nova = "A"
            elif izstop == Smer.DOL or izstop == Smer.LEVO:
                nova = "T"
        elif vstop == Smer.DOL or vstop == Smer.DESNO:
            if izstop == Smer.GOR or izstop == Smer.DESNO:
                nova = "T"
            elif izstop == Smer.DOL or izstop == Smer.LEVO:
                nova = "D"

        # Če je prejšnja oznaka N, smo na točki prvič in obvelja nova oznaka
        # po zgornjem pravilu. Sicer se določi po tabeli prehajanja stanj:

        par = prejsnja + nova

        if par == "DA" or par == "AD" or par == "TT":
            nova = "T"
        elif par == "DT" or par == "TD" or par == "DD":
            nova = "D"
        elif par == "AT" or par == "TA" or par == "AA":
            nova = "A"

        self.oznake[y_o, x_o] = nova
        return 0

    def podaj_najdaljsi_obris(self):
        """Vrne najdaljšega izmed najdenih obrisov, ki ponavadi pripada
        največjemu objektu na sliki."""
        # TODO: spiši funkcijo

        seznam_obrisov = self.obrisi
        max_index = 0
        max_dolzina = 0

        for i in range(len(seznam_obrisov)):

            if(len(seznam_obrisov[i]['kode']) > max_dolzina):
                
                max_dolzina = len(seznam_obrisov[i]['kode'])
                max_index = i

        return self.obrisi[max_index]

    def narisi_vse_obrise(self):
        """Vrne sliko z vsemi obrisi vrisanimi. Vsak obris ima na končni sliki
        drugačen sivi nivo."""

        slika_obrisov = np.zeros_like(self.slika)
        # TODO: spiši funkcijo

        seznam_obrisov = self.obrisi
        sivi_nivo = 50

        for i in range(len(seznam_obrisov)):

            sivi_nivo = sivi_nivo + 10
            for tocka in seznam_obrisov[i]['tocke']:
                slika_obrisov[tocka[1],tocka[0]] = sivi_nivo

        return slika_obrisov

def narisi_obris(slika, obris):
    """Izriše podani obris na sliko iste resolucije, kot vhodna slika."""
    slika_obrisa = np.zeros_like(slika)
    # TODO: spiši funkcijo

    for tocka in obris['tocke']:
        slika_obrisa[tocka[1],tocka[0]] = (255,255,255)

    return slika_obrisa

if __name__ == "__main__":
    # Preberi sliko s podanega datotečnega imena
    try:
        filename = sys.argv[1]
    except IndexError:
        print("Uporaba programa: python obrisi.py <datotečno ime slike>")
        sys.exit(1)

    # Branje slike in pretvorba barvnega prostora
    slika = cv2.imread(filename)
    slika = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
    sivaSlika = cv2.cvtColor(slika, cv2.COLOR_RGB2GRAY)

    # TODO: Obdelava sivinske slike, npr. s postopki
    # izravnave histograma, filtrom mediane in Gaussovim glajenjem

    obdelanaSlika = cv2.medianBlur(sivaSlika, 9)


    # Izračun in prikaz histograma ter slik

    histogram = izracunajHistogram(obdelanaSlika)
    narisiHistogram(histogram)
    
    plt.figure()
    plt.title("Barvna slika")
    plt.imshow(slika)

    plt.figure()
    plt.title("Sivinska slika")
    plt.imshow(sivaSlika, cmap="gray")

    plt.figure()
    plt.title("Obdelana slika")
    plt.imshow(obdelanaSlika, cmap="gray")

    prag = dolociPrag(obdelanaSlika)
    
    _, upragovljenaSlika = cv2.threshold(obdelanaSlika, prag, 255, 0)

    print(prag)

    plt.figure()
    plt.title("Upragovljena slika")
    plt.imshow(upragovljenaSlika, cmap="gray")

    Objekt = Iskalnik(upragovljenaSlika)

    Objekt.isci_obrise()

    max_obris_obris = Objekt.podaj_najdaljsi_obris()

    vsi_obrisi = Objekt.narisi_vse_obrise()

    plt.figure()
    plt.title("Slika obrisov")
    plt.imshow(vsi_obrisi,cmap="gray")

    # Izris najdaljsega obrisa
    slika_max_obrisa = narisi_obris(slika,max_obris_obris)
    plt.figure()
    plt.title("Slika najdaljsega obrisa")
    plt.imshow(slika_max_obrisa)

    # TODO: dopolni kodo razreda obrisi.Iskalnik
    # TODO: poišči obrise v upragovljeni sliki
    # TODO: posebej izriši vse obrise naenkrat ter najdaljši obris
    # TODO: shrani najdaljši obris v besedilno datoteko,
    #       v obliki zaporedja točk

    plt.show(block=False)

    input("Press any key.")
    plt.close("all")
    sys.exit(0)