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


def log_likelihood(X, y, theta):
    h = sigmoid(np.dot(X, theta))
    likelihood = y * np.log(h) + (1 - y) * np.log(1 - h)
    return np.sum(likelihood)


def gradient_ascent(X, y, theta, alpha, num_iterations):
    m = len(y)
    likelihood_history = []
    for iteration in range(num_iterations):
        h = sigmoid(np.dot(X, theta))
        likelihood = log_likelihood(X, y, theta)
        theta -= alpha * np.dot(X.T, (sigmoid(np.dot(X, theta)) - y))
        w_y.append(iteration + 1)
        w_0.append(theta[0, 0])
        w_1.append(theta[1, 0])
        w_2.append(theta[2, 0])
        w_3.append(theta[3, 0])
        w_4.append(theta[4, 0])

        likelihood_history.append(likelihood)
    return theta, likelihood_history


alpha = 0.5
num_iterations = 150
trained_theta, likelihood_history = gradient_ascent(X_train, y_train, theta, alpha, num_iterations)


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
plt.clf()


plt.plot(range(1, num_iterations + 1), likelihood_history, color='green')
plt.xlabel('Numărul de iterații')
plt.ylabel('Log-Likelihood')
plt.title('Convergența log-likelihood în timpul antrenării')
plt.savefig("log_likelihood.png")


print(theta)
