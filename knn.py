import numpy as np

class KNNMedianRegressor:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            # 1. Calcula distâncias euclidiana
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            print("Distâncias:", distances)

            # 2. Ordena os mais proximos
            k_nearest_indices = np.argsort(distances)[:self.k]
            print("Índices dos vizinhos mais próximos:", k_nearest_indices)

            # 3. Calcula a mediana
            median_pred = np.median(self.y_train[k_nearest_indices])
            y_pred.append(median_pred)

        return np.array(y_pred)

# Teste final
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([10, 20, 30, 40, 500])
X_test = np.array([[2.7]])

model = KNNMedianRegressor(k=3)
model.fit(X_train, y_train)
print("Resultado final:", model.predict(X_test))