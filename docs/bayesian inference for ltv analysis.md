---
marp: true
theme: default
# header: "**ヘッダータイトル1** __ヘッダータイトル2__"
# footer: "yukoga@" 
backgroundImage: url('https://marp.app/assets/hero-background.svg')

size: 16:9
paginate: true

style: |
    @import url('https://fonts.googleapis.com/css?family=Noto+Sans+JP&display=swap');
    section.title {
        font-family: 'Noto Sans JP', sans-serif;
        justify-content: center;
        text-align: left;
    }

    section {
        font-family: 'Noto Sans JP', sans-serif;
        justify-content: start;
    }

    section > bold {
        font-weight: bold;
    }

---
<!-- 
_class: title 
_paginate: false
-->
# Customer Lifetime Value modeling with JAX
&nbsp; &nbsp; yukoga@

<!--
_color: white
_footer: 'Photo by [Susan Jane Golding](https://flic.kr/p/28T85Ae)'
-->
![bg opacity:.9 brightness:0.4](43241712441_784686bd10_k.jpg)

---
## Problem statement / Motivation 

### Customer Lifetime Value (CLV / CLTV) ...  
A metric that businesses see how much profit their customers bring to them over time. 

### Challenges in existing solutions 
- Some solutions using "pesuedo" CLTV (e.g. churn probability within a specific time frame). i.e. Would like to know CLTV for "**any**" time/day.     
- Too late to take an action to customers who is likely to churn.
- Hard to incorporate customer characteristics into the model.  


---
<!--
style: |
    table {
        font-size: 18pt;
    }
    thead th {
        background-color: #DDDDDD;
        border-color: #CCCCCC;
    }
    tbody tr td:first-child {
        background-color: #EEEEEE;
        border-color: #CCCCCC;
    }
    tbody tr td:nth-child(n+2) {
        background-color: #FFFFFF;
        border-color: #CCCCCC;
    }
    tbody tr:last-child {
        background-color: rgba(0, 0, 0, 0.0);
        border-style: solid;
        border-width: 0;
    }
-->
## CLTV modeling problem's class 

|purchase behavior <br />and churn observation type|Contractual<br />(Customer 'death' can be observed)|Non contractual<br />(Customer 'death' is unobserved)|
|---|---|---|
|**Continuous**<br />(Purchases can happen at any time.)|:credit_card: Shopping with credit card|:convenience_store: Retail  <br /> :computer: ecommerce|
|**Discrete**<br />(Purchases occur at fixed periods or freqency.)|:newspaper: Subscription <br />:dollar: Insurance / Finance|:nail_care: Nail salons|
|Common methodology|Survival Analysis|BTYD model|

---
## How to model expected CLTV
For contractual model, the expectation value of CLTV can be written as follows (Fader, Peter, & Bruce (2007a)): 
$$
\begin{aligned}
E[CLTV] &= \displaystyle{\sum_{t=0}^{\infty}} \hspace{2pt}\frac{m}{(1+d)^t}s(t)
\hspace{10pt}...\hspace{10pt}(1) \\
\end{aligned}
$$
- $t$ : Discrete time.  
- $m$ : Monetize value.  
- $s(t)$ : survival function at time $t$.
- $d$ : discount rate reflecting the time value of money.


---
## How to model expected CLTV
Here's an example for the following scenario:  
- We have 1,000 customers at $t_0$(e.g. year 1), 670 at $t_1$, 480 at $t_2$, 350 at $t_3$ ...  
- $m = \$ 50/year$
- $d = 15\%$

$$
\begin{aligned}
E[CLTV] &= {\footnotesize 50} + {\footnotesize\frac{50}{1.15}\cdot\frac{670}{1000}} + {\footnotesize\frac{00}{1.15^2}\cdot\frac{480}{1000}} + {\footnotesize\frac{50}{1.15^3}\frac{350}{1000}}\hspace{5pt}...
\end{aligned}
$$

- For given observed data, we can calculate cLTV using the eq. (1) as above.  
- But the problem is, we don't have the right survival function $s(t)$ for new customers whose CLTV we're going to predict.
---
## Geometric-beta model (Fader, Peter & Bruce (2007a)) 
Assume customer lifetime follows a geometric distribution because customers can churn only once.
- Churn probability : $\theta$  
- Retention probability for customer : $1-\theta$  
- Churn probability at time $t$ :  
$$
P(T=t | \theta) = \theta(1-\theta)^{t-1} \hspace{10pt}...\hspace{10pt}(2)
$$ 

---
## Geometric-beta model (cont'd)
For given Churn and retention probability, survival rate and retention rate at time t as follows:  
- Survival rate:  
$$
s(T=t | \theta) = (1-\theta)^t \hspace{10pt}...\hspace{10pt}(3)
$$ 
We model the heterogeneity of $\theta$ as a beta distribution (Since $\theta$ is bounded between $[0, 1]$ as it's probability). 
- Prior distribution for $\theta$:  
$$
f(\theta|\alpha,\beta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)} \hspace{10pt}...\hspace{10pt}(5)
$$ 
- $\alpha$, $\beta$ : Given parameters for the prior distribution.  


---
## Geometric-beta model (cont'd)
If we get the true $\theta$ from inference using $P(T|\theta)$ and prior $f(\theta)$, we'll get LTV through $s(t)$:
$$
P(T|\theta)f(\theta) \hspace{1pt}\small{(eq. 2 \& 5)} \underset{inference}{\to} \hat{\theta} \to s(T|\hat{\theta}) \hspace{1pt} \small{(eq. 3)} \to LTV \hspace{1pt}\small{(eq. 1)}\hspace{10pt}...\hspace{10pt}(4)
$$ 

---
## Maximum Likelihood Estimation with JAX
A naive way to get inferenced  $\theta$ is Maxium Likelihood Estimation a.k.a. MLE. Funadamental code with JAX for MLE is as follows: 

```python
from jax.numpy as jnp
from jax.scipy.stats import geom

def loglikelihood(y, w):
    return jnp.sum(geom.logpmf(y, w))

def loss(y, w):
    size = y.shape[0]
    return (-1.0 * loglikelihood(y, w)) / size  # negative loglikelihood
```

---
## Maximum Likelihood Estimation with JAX (cont'd)
```python

from jaxopt import ScipyBoundedMinimize

m = ScipyBoundedMinimize(
    fun=loss,
    method="l-bfgs-b",
    options={...},
)
lb, ub = 0.00001, 0.99999
result = m.run(param, (lb, ub), (y, _))

## result.x --> parameter

## For continuous distributions, there's minimize function in jax.scipy.optimize
## result = minimize(loss, param, (y, _), method="BFGS", options={...})
```

---
## Maximum Likelihood Estimation with JAX (experiment)

Survival function observations and estimated $(\theta = 0.2)$
![MLE for uniform geom height:500](./images/survive_uniform_mle.png)

---
## Maximum Likelihood Estimation with JAX (experiment: cont'd)

LTV observations and estimated $(\theta = 0.2)$
![MLE for uniform geom height:500](./images/ltv_uniform_mle.png)

---
## Maximum Likelihood Estimation with JAX (experiment: cont'd)

Survival function observations and estimated $(Geom(\theta=0.2) + Poi(\lambda=1.0))$
![MLE for poisson & geom height:500](./images/survive_poisson_noise_mle.png)

---
## Maximum Likelihood Estimation with JAX (experiment: cont'd)

LTV observations and estimated $(Geom(\theta=0.2) + Poi(\lambda=1.0))$
![MLE for poisson & geom height:500](./images/ltv_poisson_noise_mle.png)

---
## Maximum Likelihood Estimation with JAX (experiment: cont'd)
Survival function observations and estimated for Mixed distributions
$(Geom(\theta=0.7) + Geom(\theta=0.1))$
![MLE for mixed geom height:500](./images/survive_mixed_geom_mle.png)

---
## Maximum Likelihood Estimation with JAX (experiment: cont'd)
LTV observations and estimated for Mixed distributions
$(Geom(\theta=0.7) + Geom(\theta=0.1))$
![MLE for uniform geom height:500](./images/ltv_mixed_mle.png)

---
## Maximum Likelihood Estimation with JAX (experiment: cont'd)
Survival observations and estimated for Mixed distributions
$(Geom(\theta=0.7) + Geom(\theta=0.1))$
![MLE for uniform geom height:500](./images/survive_mixed_geom_segmented_mle.png)

---
## Bayesian inference with JAX
We need to estimate the heterogeneity of $\theta$ based on given data accordingly. Bayesian inference will work for that case. Assume a beta distribution as a prior of $\theta$ (Since $\theta$ is bounded between $[0, 1]$ as it's probability). 

- Prior distribution for $\theta_{i}$ :  
$$
f(\theta_{i}|\alpha,\beta) = \frac{\theta_{i}^{\alpha-1}(1-\theta_{i})^{\beta-1}}{B(\alpha,\beta)} \hspace{10pt}...\hspace{10pt}(5)
$$ 
- $\alpha$, $\beta$ : Latent parameters contains customer's characteristics as follows:  

---
<!--
style: |
    table {
        font-size: 18pt;
    }
    thead th {
        background-color: #DDDDDD;
        border-color: #CCCCCC;
    }
    tbody tr td:first-child {
        background-color: #EEEEEE;
        border-color: #CCCCCC;
    }
    tbody tr td:nth-child(n+2) {
        background-color: #FFFFFF;
        border-color: #CCCCCC;
    }
    tbody tr:last-child {
        background-color: rgba(0, 0, 0, 0.0);
        border-style: solid;
        border-width: 0;
    }
-->
## Bayesian inference with JAX (experiment: cont'd)
Training data for Bayesian analysis as follows: 

||Segment (= feature)|Churn date (= target)|
|---|---|---|
|1|A|1|
|2|B|7|
|3|A|3|
|4|B|5|
|..|...|...|

---
## Bayesian inference with JAX (experiment: cont'd)
Here is PyMC4 code to get inferenced  $\theta$. Funadamental code with PyMC4 for baysian inference is as follows: 

```python
import pymc as pm

model = pm.Model()

with model:
    x_ = pm.Data('features', X_train.values, mutable=True)
    theta = pm.Beta('theta', 1., 1., shape=[X_train.max()+1])
    obs = pm.Geometric('obs', theta[x_], observed=y_train.values)
    idata = pm.sample(draws=2000, target_accept=0.9)

```
---
## Bayesian inference with JAX (experiment: cont'd)
Here is PyMC4 code to get inferenced  $\theta$. Funadamental code with PyMC4 for baysian inference is as follows: 

```python
import arviz as az

az.plot_trace(idata, ['theta'])
```
![Bayes mixed geom](./images/pymc_trace.png)

---
## Bayesian inference with JAX (experiment: cont'd)
Survival observations and estimated for Mixed distributions
$(Geom(\theta=0.7) + Geom(\theta=0.1))$
![bayes for mixed geom height:500](./images/survive_mixed_geom_bayes.png)

---
## Bayesian inference with JAX (experiment: cont'd)
LTV observations and estimated for Mixed distributions
$(Geom(\theta=0.7) + Geom(\theta=0.1))$
![bayes for mixed geom height:500](./images/ltv_mixed_bayes.png)

---
<!--
_paginate: false
-->
<style scoped>
h2 {
    color: rgba(255, 255, 255, 0.65);
    font-size: 200%;
}
</style>
## Thanks. 


<!--
_footer: 'Photo by [Tobi Gaulke](https://www.flickr.com/photos/gato-gato-gato/45025977691)'
-->
![bg opacity:.9 brightness:0.8](45025977691_0103ce74f0_k.jpg)
