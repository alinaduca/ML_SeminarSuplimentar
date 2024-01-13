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
theta += 1
w_y = []
w_0 = []
w_1 = []
w_2 = []
w_3 = []
w_4 = []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_hessian(X, theta):
    m = len(X)
    hessian = np.zeros((len(theta), len(theta)))
    for i in range(m):
        xi = X[i, :].reshape(-1, 1)
        sigmoid_i = sigmoid(np.dot(theta.T, xi))
        hessian += sigmoid_i * (1 - sigmoid_i) * np.dot(xi, xi.T)
    return hessian


def newton_raphson(X, y, theta, num_iterations):
    likelihood_history = []
    for iteration in range(num_iterations):
        gradient = np.dot(X.T, (sigmoid(np.dot(X, theta)) - y)) * 0.3
        hessian = compute_hessian(X, theta)
        theta -= np.linalg.inv(hessian).dot(gradient)
        probabilities = sigmoid(np.dot(X, theta))
        likelihood = np.sum(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
        likelihood_history.append(likelihood)
        w_y.append(iteration + 1)
        w_0.append(theta[0, 0])
        w_1.append(theta[1, 0])
        w_2.append(theta[2, 0])
        w_3.append(theta[3, 0])
        w_4.append(theta[4, 0])
    return theta, likelihood_history


num_iterations = 10
trained_theta, likelihood_history = newton_raphson(X_train, y_train, theta, num_iterations)
np.set_printoptions(precision=4, suppress=True)
print(trained_theta)


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
plt.xticks(range(1, num_iterations + 1, 1))
plt.title('Graficul ponderilor')
plt.legend()
plt.savefig("ponderi.png")
plt.clf()


plt.xticks(range(1, num_iterations + 1, 1))
plt.plot(range(1, num_iterations + 1), likelihood_history, color='green')
plt.xlabel('Numărul de iterații')
plt.ylabel('Log-Likelihood')
plt.title('Convergența log-likelihood în timpul antrenării')
plt.savefig("log_likelihood.png")
