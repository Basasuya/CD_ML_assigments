## 1. Machine Learning Problems

(a)

1. BF

2. C
3. BD
4. BG
5. AE
6. AD
7. BF
8. AE
9. BF

(b)

False. You should have the test dataset and the train dataset. Because the training on the training set will cause overfit, if you do not distinguish between the test set and the training set, the final result will be false high.

## 2. Bayes Decision Rule

(a)

1. $P(B_{1} = 1)=\frac{1}{3}$ 
2. $P(B_{2}=0|B_{1}=1)=1$
3. $P(B_{1}=1|B_{2}=0)=\frac{1}{2}$
4. Either choises have the same probility



## 3. Gaussian Discriminant Analysis and MLE

(a)

$\displaystyle P(y=1|x)=\frac{e^{-\frac{1}{2}x_{1}^{2}x_{2}^{2}}} { e^{-\frac{1}{2}x_{1}^{2}x_{2}^{2}} + e^{-\frac{1}{2}(x_{1}-1)^{2}(x_{2}-1)^{2}}}$



Discisoin Boudry: $x_1^{2}*x_2^2=(x_1-1)^2(x_2-1)^2$

np.linalg.det