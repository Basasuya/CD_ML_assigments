* Recognize handwritten digits by looking for the most similar image in a large dataset of labeled digit images, then use its label as result  (Supervised Learning classification)
* Represent image as a well chosen 64 bits integer, so that similar images will be
  represented as integers with small hamming distance(Unsupervised Learning Dimension Reduction)

**Bayesian**: prior: $p(w_1) > p(w_2)$  Likelihood: $p(x | w_1) > p (x | w_2)$  Posterior:  $p(w_i|x)=\frac{p(x|w_{i})P(w_i)}{P(x)}$

Posterior = (Likelihood 􏲈 * Prior) / Evidence

Bayes Risk: $R(a_i|x)=\sum_{j=1}^{c} \lambda(a_i|w_j)P(w_j|x)$ selcet the minimum Risk 

if $\lambda(a_i|w_j)= 0,1$ then $R(a_i|x) =1-P(w_i|x)$ we can choose the minmum  $p(w_i|x)$

we can use normal distribution to estimate $P(x|w_i)$

$\displaystyle{ P(x|w_i) \sim N(u_i, \phi_{i}) =\frac{1}{(2\pi^{d/2}|\phi|^{1/2})}exp[-1/2(x-u)^T\phi^{-1}(x-u)]}$

$u=1/n\sum_{k=1}^{n}x_k \quad \phi^{2}=1/n\sum_{k=1}^{n}(x_k-u)^{2}$

naive bayes: $p(w|x_1,...x_p) \propto p(x1,...x_p|w)p(w) = p(x_1|w)...p(x_p|w) \quad P(x_i|w_k)=|x_{ik}+1|/(N_{wk}+K)$

the numerical type we can use $P(x_i|w_j)=1/\sqrt{2\pi\phi^{2}_{ij}}exp(-(x_i-u_{ij}^{2}/ 2\phi_{ij}^{2}))$

**Perception**：if the model predicts x correct, do nothing else $w=w_{t-1}+xy$

if w is a solution, for any x in the training set, we have $w^Txy \ge0$

In conclusion, a linear discriminant function divides the feature space by a hyperplane decision surface$g(x)=w^{t}x+w_0$,The orientation of the surface is determined by the normal vector w and the location of the surface is determined by the bias, the distance hyperplane is $r=g(x)/||w||$

$E(x)=-\sum_{i} w^{T}x_{i}y_{i}$

**Linear Regression**: mean‐squared error(MSE): $E(x)=1/n\sum_{i=1}^{n}(y_i-f(x_i))^2$  the gradient descent: $\delta E(x) =-2X(y-X^{T}a) \quad a_{k+1}=a_k - \lambda 2X(X^{T}a-y)$ 

Ridge regression(Regularized) $E(x)=\sum_{i=1}^{n}(y_{i}-x_{i}^{T}a)^{2}+\lambda \sum_{i=1}^{p}a_{i}^{2}$

if we use matrix solution when $XX^T$is not singular $a=(XX^{T})^{-1}Xy \quad a=(XX^{T}+\lambda I)^{-1}Xy$

LASSO: $E(x)=\sum_{i=1}^{n}(y_{i}-x_{i}^{T}a)^{2}+\lambda \sum_{i=1}^{p}a_{i}$ L1范数比L2更容易得到稀疏解 ，更少的非零解

<img src="../Screen Shot 2019-01-23 at 2.07.22 PM.png" width=400px><img src="../Screen Shot 2019-01-23 at 2.10.49 PM.png" width=400px>

**logistic Regression**: $E(x)=\sum log(1+e^{-y_ia^{T}x_{i}})$ another equation: $E(x)=-1/m \sum_{i=1}^{m}(y_ilog(h(x_i)+(1-y_i)log(1-h(x_i))))$   $a=a^{'}-\sum_{i=1}^{m}(h(x_i)-y_i)x_i$

**SVM**: maximum the Geometrical Margin $\lambda = y\frac{w^{T}x+b}{||w||}$let$y(w^Tx+b)=1$ 

<img src="../Screen Shot 2019-01-23 at 2.41.05 PM.png" width=400px><img src="../Screen Shot 2019-01-23 at 4.34.39 PM.png" width=400px>

so it is the **Hinge loss**: $min_{w, b}\{\sum_{i=1}^{n}max[1-y(w^{T}x_{i}+b, 0)]+1/2C||w||^{2}\}$ other loss such as square loss, logistic loss add Slack variables: $min_{w,b}1/2||w||+C\sum\phi_{i} \quad y(w^{T}x_i+b) \ge 1 - \phi_{i} \quad \phi i \ge 0$ 

**Kernels:** $f(x)=\sum a_{i}k(x_{i}, x)$linear kernel $x^{T}x^{'}$ Polynomial kernels$(x^{T}x^{'}+1)^d$ RBF kernel$exp(-||x-x^{'}||^{2}/2\phi^{2})$

**neural networks**: relu, tanh, dropout, backpropagation Batch learning vs online learning(such as Stochastic) the sigmoid function $\sigma (x)=1/(1+e^{-x}) \quad \sigma(x)^{'}=(1-\sigma(x))\sigma(x)$ $32x32x3 - 5x5x3 = 28*28$ use $5*5*3 + 1=76$parameters

**Knn**: small k high bias low variance, large k, large k low bias high varince

**DecisionTree**: $H(x)=-\sum p_ilog(p_i) \quad H(D | A) = \sum_{i=1}^{n}|D_i|/|D|H(D_i)$

**RandomForest**: bagging aggregating low bias(not necessary) high varience(should be different)

spectral clustering**:    Points assigned to same cluster should be highly similar. Points assigned to different clusters should be highly dissimilar. d = np.sum(W, axis=1)   l= np.diag(d) - W s = np.diag(1.0 / (d ** (0.5))) result =  np.dot(np.dot(s, l), s)  lam, H = np.linalg.eig(result) F = np.take(H,lam.argsort()[:k], axis=-1)  idx = kmeans(F, k)

<img src="../Screen Shot 2019-01-23 at 4.47.06 PM.png" width=400px><img src="../Screen Shot 2019-01-23 at 5.11.52 PM.png" width=400px>

**PCA**: 最近重构性：样本点到这个超平面的距离足够近，最大可分性到超平面的距离尽可能分开

$\sum_{i=1}^{m}||\sum_{j=1}^{d}z_{ij}w_{j}-x_{i}||^2=\sum_{i=1}^{m}z_i^Tz_i-2\sum_{i=1}^{m}z_i^TW^Tx_i+const \quad \max tr(W^TXX^TW) s.t. WW^T=I$

$XX^TW=\lambda W$

**LDA**: Find a transformation 􏰁, such that the 􏰁􏰇􏱒􏰣 and 􏰁􏰇􏱒􏰎 are maximally separated & each class is minimally dispersed (maximum separation) **LPP** Unsupervised, but it is very easy to have supervised (semi‐
supervised) extensions. **LDA**: supervised learning

**topic modeling**: plsa $P(w | d)=n(d, w)/\sum n(d, w^{'}) \quad P(w | d)=\sum P(w | z)P(z | d)$

**Matrix Factorization**: Singular Value Decomposition，Nonnegative Matrix Factorization， Sparse Coding