import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = {
    'NotHeavy': [1, 1, 0, 0, 1, 1, 1, 0],
    'Smelly': [0, 0, 1, 0, 1, 0, 0, 1],
    'Spotted': [0, 1, 0, 0, 1, 1, 0, 0],
    'Smooth': [0, 0, 1, 1, 0, 1, 1, 0],
    'Edible': [1, 1, 1, 0, 0, 0, 0, 0]
}
df_train = pd.DataFrame(data_train)
df_train.insert(0, 'Bias', 1)
X_train = df_train.iloc[:, :-1].values  # ia totul pana la ultima coloana
y_train = df_train.iloc[:, -1].values.reshape(-1, 1)  # ia doar ultima coloana
theta = np.zeros((X_train.shape[1], 1))
w_y = []
w_0 = []
w_1 = []
w_2 = []
w_3 = []
w_4 = []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))   # sigma(x_i) = 1/(1+e^(-x_i))


def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = (-1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))  # funcția de verosimilitate condiționată
    return J


def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    J_history = []
    for iteration in range(num_iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (h - y)) / m
        theta -= alpha * gradient
        w_y.append(iteration + 1)
        w_0.append(theta[0, 0])
        w_1.append(theta[1, 0])
        w_2.append(theta[2, 0])
        w_3.append(theta[3, 0])
        w_4.append(theta[4, 0])
        cost = cost_function(X, y, theta)[0, 0]
        J_history.append(cost)
        # if (iteration + 1) % 100 == 0:
            # print('Costul pentru iteratia ' + str(iteration + 1) + ' este ' + str(cost))
    return theta, J_history


alpha = 0.5
num_iterations = 2000
trained_theta, J_history = gradient_descent(X_train, y_train, theta, alpha, num_iterations)

plt.plot(range(1, num_iterations + 1), J_history, color='blue')
plt.xlabel('Numărul de iterații')
plt.ylabel('Cost')
plt.title('Convergența costului în timpul antrenării')
plt.savefig("regression.png")
plt.clf()


data_test = {
    'NotHeavy': [0, 1, 1],
    'Smelly': [1, 1, 1],
    'Spotted': [1, 0, 0],
    'Smooth': [1, 1, 0]
}

df_test = pd.DataFrame(data_test)
df_test.insert(0, 'Bias', 1)
X_test = df_test.values


def predict(X, theta):
    return sigmoid(np.dot(X, theta))


plt.xlabel('Numărul de iterații')
plt.ylabel('Valorile ponderilor')
predictions = predict(X_test, trained_theta)
predicted_classes = (predictions >= 0.5).astype(int)
for i, pred_class in enumerate(predicted_classes):
    print(f'Ciuperca {chr(ord("U") + i)} este comestibilă: {bool(pred_class[0])}')


plt.plot(w_y, w_0, marker='^', label='w_0')
plt.plot(w_y, w_1, marker='*', label='w_1')
plt.plot(w_y, w_2, marker='+', label='w_2')
plt.plot(w_y, w_3, marker='H', label='w_3')
plt.plot(w_y, w_4, marker='.', label='w_4')
plt.title('Graficul ponderilor')
plt.legend()
plt.savefig("ponderi.png")
