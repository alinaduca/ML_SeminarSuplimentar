# 📈 ML_SeminarSuplimentar

This repository includes implementations of the sigmoidal perceptron, logistic regression using the gradient ascending method and logistic regression using the Newton-Raphson method, made for the additional Machine Learning seminar.
 
The dataset used is the one from the mushroom problem, exercise 4.2 on page 470 of the [Machine Learning Exercise Collection](https://profs.info.uaic.ro/~ciortuz/ML.ex-book/editia-2023f/ex-book.20sept2023.pdf):

<img align="center" src="https://github.com/alinaduca/ML_SeminarSuplimentar/blob/main/dataset.png">

Some of these graphs can be found in the presentation of the [Regression chapter](https://profs.info.uaic.ro/~ciortuz/ML.ex-book/SLIDES/ML.ex-book.SLIDES.Regression.pdf) presented in the Machine Learning course at the Faculty of Informatics of the "Al. I. Cuza" University in Iasi, academic year 2023-2024.


## Results and graphs

1. Sigmoidal perceptron - uses gradient descent to calculate the minimum value of the conditional probability:

<img align="center" src="https://github.com/alinaduca/ML_SeminarSuplimentar/blob/main/sigmoidal-perceptron/log_likelohood.png">

<img align="center" src="https://github.com/alinaduca/ML_SeminarSuplimentar/blob/main/sigmoidal-perceptron/ponderi.png">

Final results (for 2000 iterations):

```
Mushroom U is edible: False
Mushroom V is edible: False
Mushroom W is edible: True
```

2. Logistic regression (ascending gradient method) - uses the ascending gradient to calculate the maximum value of the conditional probability:

<img align="center" src="https://github.com/alinaduca/ML_SeminarSuplimentar/blob/main/logistic-regression-gradient/log_likelohood.png">

<img align="center" src="https://github.com/alinaduca/ML_SeminarSuplimentar/blob/main/logistic-regression-gradient/ponderi.png">

Final results (for 150 iterations):

```
Mushroom U is edible: False
Mushroom V is edible: False
Mushroom W is edible: True

 w_0 = 0.75287357
 w_1 = 0.00332853
 w_2 = -0.75367651
 w_3 = -0.75768382
 w_4 = -1.51094581
```

2. Logistic regression (Newton-Raphson method) - uses the Hessian matrix and the gradient vector to calculate the maximum value of the conditional probability:

<img align="center" src="https://github.com/alinaduca/ML_SeminarSuplimentar/blob/main/logistic-regression-newton-raphson/log_likelohood.png">

<img align="center" src="https://github.com/alinaduca/ML_SeminarSuplimentar/blob/main/logistic-regression-newton-raphson/ponderi.png">

Final results (for 10 iterations):

```
Mushroom U is edible: False
Mushroom V is edible: False
Mushroom W is edible: True

 w_0 = 0.7565
 w_1 = -0.0002
 w_2 = -0.7563
 w_3 = -0.7563
 w_4 = -1.5126
```
