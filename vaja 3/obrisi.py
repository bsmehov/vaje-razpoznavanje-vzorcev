from upragovljanje import *
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
        dolzine = []
        for obris in self.obrisi:
            dolzina = len(obris["kode"])
            dolzine.append(dolzina)
        return self.obrisi[np.argmax(dolzine)]

    def narisi_vse_obrise(self):
        """Vrne sliko z vsemi obrisi vrisanimi. Vsak obris ima na končni slii
        drugačen sivi nivo."""
        slike = []

        for i in range(len(self.obrisi)):
            slika = narisi_obris(self.slika, self.obrisi[i])
            slike.append(slika * (i + 1))

        koncna_slika = slike[0]
        for slika in slike[1:]:
            koncna_slika += slika

        if koncna_slika.max() < 32:
            koncna_slika *= 8
        elif koncna_slika.max() < 64:
            koncna_slika *= 4
        elif koncna_slika.max() < 128:
            koncna_slika *= 2

        return koncna_slika

def narisi_obris(slika, obris):
    """Izriše podani obris na sliko iste resolucije, kot vhodna slika."""
    slika_obrisa = np.zeros((slika.shape[0], slika.shape[1]), 
                            dtype="uint8")

    # točke so podane v koordinatah (x, y), medtem, ko so pri koordinatah
    # numpy arrayev na prvem indeksu vrstice. Zato jih obrnemo.
    slika_obrisa[obris["ZT"][::-1]] == 1

    for tocka in obris["tocke"]:
        slika_obrisa[tocka[::-1]] = 1
    return slika_obrisa

    
