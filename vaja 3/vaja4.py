from znacilke import *

if __name__ == "__main__":
    if "X.npy" in os.listdir(".") and "y.npy" in os.listdir("."):
        X, y = nalozi_znacilke()
    else:
        X, y = shrani_znacilke()

    accs = navzkrizno_preverjanje(X, y)
    print(accs)
    print(np.mean(accs))
    sys.exit(0)
    X, y = premesaj_vrstice(X, y)

    X_test = X[:72]
    X_train= X[72:]
    y_test = y[:72]
    y_train= y[72:]

    razpoznavalnik = CSMKNN(mera=evk, nacin="min")
    razpoznavalnik.fit(X_train, y_train)

    y_hat = razpoznavalnik.predict(X_test)

    # apriorne verjetnosti razredov
    p_i = [np.mean(y_test == i) for i in range(5)]

    conf_mat = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            p_ij = np.mean(np.logical_and(y_test == i, y_hat == j))
            conf_mat[i, j] = p_ij / p_i[i]

    plt.imshow(conf_mat)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
