from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from loadFaces import load_faces
import numpy as np
from sklearn.preprocessing import normalize

# Load Data Face Sets
dataset_faces = load_faces('centered_faces')
X, y = [dataset_faces[0], dataset_faces[1]]

# Normalizar dataset
X = normalize(X)

Nr = 100
hitsVector = []

for i in range(Nr):
    print('Rodada : {} '.format(i + 1))
    # 1- Metodo de separacao ordenada

    # X_train = []
    # X_test = []
    # y_train = []
    # y_test = []

    # Xi = []
    # yi = []
    # ant_i = 0

    # for i in range(len(y)):

    #     if (y[i] == y[ant_i])and(i != (len(y) - 1)):
    #         Xi.append(X[i])
    #         yi.append(y[i])
    #     elif (i == (len(y) - 1)):
    #         Xi.append(X[i])
    #         yi.append(y[i])
    #         X_t1, X_t2, y_t1, y_t2 = train_test_split(Xi, yi, test_size=0.2)
    #         X_train.append(X_t1)
    #         X_test.append(X_t2)
    #         y_train.append(y_t1)
    #         y_test.append(y_t2)
    #         Xi = []
    #         yi = []
    #     else:
    #         X_t1, X_t2, y_t1, y_t2 = train_test_split(Xi, yi, test_size=0.2)
    #         X_train.append(X_t1)
    #         X_test.append(X_t2)
    #         y_train.append(y_t1)
    #         y_test.append(y_t2)
    #         Xi = []
    #         yi = []
    #         Xi.append(X[i])
    #         yi.append(y[i])

    #     ant_i = i

    # X_train = np.reshape(
    #     X_train, (len(X_train[0]) * len(X_train), len(X_train[0][0]))).tolist()
    # X_test = np.reshape(
    #     X_test, (len(X_test[0]) * len(X_test), len(X_test[0][0]))).tolist()

    # y_train = np.reshape(y_train, len(y_train[0]) * len(y_train)).tolist()
    # y_test = np.reshape(y_test, len(y_test[0]) * len(y_test)).tolist()

    # Fim do metodo 1

    # 2 - Metodo de separacao aleatoria

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fim do metodo 2

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    count = 0
    predict = neigh.predict(X_test)
    hits = 0

    for index in y_test:

        #print('Actual: {} -> Predict {}'.format(index, predict[count]))

        if(index == predict[count]):
            hits = hits + 1

        count = count + 1

    # print('Accuracy :{}/{} -> {}%'.format(hits,len(y_test), (hits / len(y_test) * 100)))

    hitsVector.append((hits / len(y_test)) * 100)

#print("Vetor de acertos: {}".format(hitsVector))
print("Mediana de acertos :{} %".format(np.median(np.array(hitsVector))))
print("Media de acertos :{} %".format(np.mean(np.array(hitsVector))))
print("Minimo acerto :{} %".format(np.min(np.array(hitsVector))))
print("Maximo acerto :{} %".format(np.max(np.array(hitsVector))))
