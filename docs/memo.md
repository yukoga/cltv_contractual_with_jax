# Geometric Distribution MLE
Ref. [Geometric Distribution MLE - CopyProgramming](https://copyprogramming.com/howto/geometric-distribution-mle#mle-for-p-in-geometric-distribution-from-exponential-distribution-two-methods-two-results)

Given data $X_1,...,X_n \sim \text{IID Geom}(\theta)$ (using the version of the geometric distribution where the value is the count of trials including the first success, so that $X_i \geqslant 1$) you have log-likelihood function:

$$
\begin{equation}
\begin{aligned}
\ell_\mathbf{x}(\theta) &= \sum_{i=1}^n 
\Big[ \ln (\theta) + (x_i-1) \ln (1-\theta) \Big] \\
&= n \ln \theta + \ln (1-\theta) \sum_{i=1}^n (x_i-1) \\
&= n \ln \theta + n (\bar{x}-1) \ln (1-\theta) \quad \quad \text{for all } 0 \leqslant \theta \leqslant 1. \\ 
\end{aligned} 
\end{equation}
$$

## Question

I am trying to prove that $\dfrac {1}{\bar{x}}$ (as per the result in this youtube clip https://www.youtube.com/watch?v=0TSMugiWPc0) is certainly the MLE for geometric distribution.  
When I try to confirm by then taking the second order condition. My calculation is that it is not less that zero! I am unsure how to write formulas on here to show what I have done, this is why I posted the clip, as it will show the first derivative.  

## Solution:

Given data $X_1,...,X_n \sim \text{IID Geom}(\theta)$ (using the version of the geometric distribution where the value is the count of trials including the first success, so that $X_i \geqslant 1$) you have log-likelihood function:

$$
\begin{equation}
\begin{aligned}

\ell_\mathbf{x}(\theta)
&= \sum_{i=1}^n \Big[ \ln (\theta) + (x_i-1) \ln (1-\theta) \Big] \\
&= n \ln \theta + \ln (1-\theta) \sum_{i=1}^n (x_i-1) \\
&= n \ln \theta + n (\bar{x}-1) \ln (1-\theta) \quad \quad \text{for all } 0 \leqslant \theta \leqslant 1. \\

\end{aligned}
\end{equation}
$$


For all $0 <\theta <1$ the first and second derivatives are:

$$
\begin{equation}
\begin{aligned}

\frac{d \ell_\mathbf{x}}{d\theta}(\theta)
&= \frac{n}{\theta} - \frac{n(\bar{x}-1)}{1-\theta} = \frac{n(1-\bar{x}\theta)}{\theta (1-\theta)}, \\
\frac{d^2 \ell_\mathbf{x}}{d\theta^2}(\theta)
&= - \frac{n}{\theta^2} - \frac{n(\bar{x}-1)}{(1-\theta)^2}
= - \frac{n(1-2\theta + \bar{x}\theta^2)}{\theta^2 (1-\theta)^2} <0.

\end{aligned}
\end{equation}
$$

We can see from the second derivative that the log-likelihood function is strictly concave. Hence, the maximising value occurs at the unique critical point $\hat{\theta} = 1/ \bar{x}$. Substitution of this value into the second derivative function yields the curvature:


$$
\frac{d^2 \ell_\mathbf{x}}{d\theta^2}(\hat{\theta})
= - \frac{n \bar{x}^3}{ \bar{x}-1 } <0.
$$


## MLE for $p$ in Geometric distribution from Exponential distribution (two methods, two results)  

### Question:
Let $Y_n$ given as $\mathrm{ceil}(X_n)$ , where $\mathrm{ceil}(x):=$ the least integer greater than or equal to $x$ and $(X_n)$ is a sequence of iid random variables from $\mathrm{Exp}(\theta),~\theta>0$ .

Then $Y_n\sim Geo(p)$ , where $p=p(\theta):=1-e^{-\theta}$ and since the maximum likelihood estimator (mle) for $\theta$ is given as $\frac{1}{\overline{X}}$ , the mle for $p(\theta)$ is $1-e^{-\frac{1}{\overline{X}}}$ .

If we compute directly the mle for $p$ using $\mathbb{P}(Y_1=y)=(1-p)^{y-1}p$ for $y\in \mathbb{N}$ , we get that the maximum likelihood estimator for $p$ is given as $\frac{1}{\overline{Y}}=\frac{1}{\overline{\mathrm{ceil}(X)}} $ , which is not the same as the previous result.

Is there some contradiction in these two results, or some fallacy?

Thank for the help.


### Solution:

As Math-fun says, you are in effect using two different sets of information, one unrounded and the other rounded up, so you should not expect the same result.

For example,

if you see the data $0.1, 0.2, 0.3, 1.1, 2.1$ then $1-e^{-1/\overline{x}} \approx 0.732$ while $\frac{1}{\overline{\lceil x \rceil}}=\frac{5}{8}= 0.625$ .
if you see the data $0.7, 0.8, 0.9, 1.9, 2.9$ then $1-e^{-1/\overline{x}} \approx 0.501$ while $\frac{1}{\overline{\lceil x \rceil}}=\frac{5}{8}= 0.625$ again since you round up to the same integers.  

Here is some R code to simulate a thousand samples of size $10$ . From the chart below you can reasonably conclude:

- the two maximum likelihood estimators for $p$ give slightly different estimates in practice, largely due to the particular sample data effects on the two expressions
- the estimators are usually closer to each other than they are to the actual parameter used in the simulation ( $0.3$ in this case)
- the estimator using the ceiling function will only give discrete values; in retrospect this is obvious looking at the formula, as it can only be the sample size divided by an integer

``` R
set.seed(2021)
n <- 10
p <- 0.3
theta <- -log(1-p)
Xdat <- matrix(rexp(n*10^3, rate=theta), ncol=n)
MLE_p_exp <- 1 - exp(-1/rowMeans(Xdat))
MLE_p_geo <- 1 / rowMeans(ceiling(Xdat))
plot(MLE_p_exp, MLE_p_geo, xlim=c(0,0.8), ylim=c(0,0.8), pch=3)
abline(0,1, col="red")
```

![](./SHppr.png)
