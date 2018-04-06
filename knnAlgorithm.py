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

#for i in range(Nr):
 #   print('Rodada : {} '.format(i + 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fim do metodo 2

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

count = 0
predict = neigh.predict(X_test)
hits = 0

for index in y_test:

    print('Actual: {} -> Predict {}'.format(index, predict[count]))

    if(index == predict[count]):
        hits = hits + 1

    count = count + 1

    # print('Accuracy :{}/{} -> {}%'.format(hits,len(y_test), (hits / len(y_test) * 100)))

    hitsVector.append((hits / len(y_test)) * 100)

#print("Vetor de acertos: {}".format(hitsVector))
#print("Mediana de acertos :{} %".format(np.median(np.array(hitsVector))))
#print("Media de acertos :{} %".format(np.mean(np.array(hitsVector))))
#print("Minimo acerto :{} %".format(np.min(np.array(hitsVector))))
#print("Maximo acerto :{} %".format(np.max(np.array(hitsVector))))
