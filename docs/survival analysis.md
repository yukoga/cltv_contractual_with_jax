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
- $m = \$ 5,000/year$
- $d = 15\%$

$$
\begin{aligned}
E[CLTV] &= {\footnotesize 5000} + {\footnotesize\frac{5000}{1.15}\cdot\frac{670}{1000}} + {\footnotesize\frac{5000}{1.15^2}\cdot\frac{480}{1000}} + {\footnotesize\frac{5000}{1.15^3}\frac{350}{1000}}\hspace{5pt}...
\end{aligned}
$$

- For given observed data, we can calculate cLTV using the eq. (1) as above.  
- But the problem is, we don't have the right survival function $s(t)$ for new customers whose CLTV we're going to predict.
---
## Geometric-beta model (Fader, Peter & Bruce (2007a)) 
Assume customer lifetime follows a geometric distribution because customers can churn only once.
- Churn probability for customer $u_i$: $\theta_{u_i}$  
- Retention probability for customer $u_i$: $1-\theta_{u_i}$  
- Churn probability for customer $u_i$ at time $t$:  
$$
P(T=t | \theta_{u_i}) = \theta_{u_i}(1-\theta_{u_i})^{t-1} \hspace{10pt}...\hspace{10pt}(2)
$$ 

---
## Geometric-beta model (cont'd)
For given Churn and retention probability, survival rate and retention rate at time t for customer $u_i$ as follows:  
- Survival rate:  
$$
s(T=t | \theta_{u_i}) = (1-\theta_{u_i})^t \hspace{10pt}...\hspace{10pt}(3)
$$ 
- Retantion rate:
$$
r(T=t) = \frac{s(t)}{s(t-1)} \hspace{10pt}...\hspace{10pt}(4)
$$ 


---
## Geometric-beta model (cont'd)
We model the heterogeneity of $\theta$ as a beta distribution (Since $\theta$ is bounded between $[0, 1]$ as it's probability). 
- Prior distribution for $\theta_{u_i}$:  
$$
f(\theta_{u_i}|\alpha_{u_i},\beta_{u_i}) = \frac{\theta_{u_i}^{\alpha_{u_i}-1}(1-\theta_{u_i})^{\beta_{u_i}-1}}{B(\alpha_{u_i},\beta_{u_i})} \hspace{10pt}...\hspace{10pt}(5)
$$ 
- $\alpha_{u_i}$, $\beta_{u_i}$ : Latent parameters contains customer's characteristics as follows:  
$$
\theta_{u_i} = {\displaystyle \sum_{k}} w_k x_k + \epsilon \hspace{10pt}...\hspace{10pt}(6)
$$ 
- $x_k$ : k-th feature for a customer. 

---
## Geometric-beta model (cont'd)
We can combine the geometric distribution (given at $eq. (1)$) and the beta distribution ($eq. (5)$) to get the joint distribution as follows: 
$$

P(T=t|\alpha_{u_i},\hspace{2pt} \beta_{u_i}) = 
    \int_{0}^{1} p(T=t|\theta_{u_i}) f(\theta_{u_i}|\alpha_{u_i}, \hspace{2pt} \beta_{u_i}) d\theta_{u_i} \hspace{10pt}...\hspace{10pt}(7) \\[8pt]
p(T=1|\alpha_{u_i}, \hspace{2pt} \beta_{u_i}) = 
    {\small\frac{B(\alpha_{u_i}+1, \beta_{u_i})}{B(\alpha_{u_i}, \beta_{u_i})} =
    \frac{\alpha_{u_i}}{\alpha_{u_i} + \beta_{u_i}}} \hspace{10pt}...\hspace{10pt}(8) \\[8pt]
p(T=t|\alpha_{u_i},\hspace{2pt} \beta_{u_i}) = {\small\frac{\beta_{u_i}+t-2}{\alpha_{u_i}+\beta_{u_i}+t-1}p(T=t-1|\alpha_{u_i},\hspace{2pt} \beta_{u_i})\hspace{5pt} (for\hspace{2pt} t>1)}\hspace{10pt}...\hspace{10pt}(9) \\[8pt]
r(t) = {\small\frac{\beta_{u_i}+t-1}{\alpha_{u_i}+\beta_{u_i}+t-1}} \hspace{10pt}...\hspace{10pt}(10)
$$ 

---
## Geometric-beta model (cont'd)
Log-Likelihood for $P(T=t|\alpha_{u_i}, \hspace{2pt}\beta_{u_i})$ can be defined as follows:

$$
L(\alpha_{u_i}, \hspace{2pt}\beta_{u_i}) = {\displaystyle \prod_t P(T=t|\alpha_{u_i}, \hspace{2pt}\beta_{u_i})} \hspace{10pt}...\hspace{10pt}(11)\\[5pt]
\log L = {\displaystyle \sum_{i}}L(\alpha_{u_i},\hspace{2pt} \beta_{u_i}) = {\displaystyle \sum_{i}\sum_{t}}P(T=t|\alpha_{u_i},\hspace{2pt} \beta_{u_i}) \hspace{10pt}...\hspace{10pt}(12)
$$ 
We need to maximize the log-likelihood (12) (or minimize negative log-likelihood) to find optimal parameters $\alpha_{u_i}$ and $\beta_{u_i}$ (i.e. set of parameters $\{w_{u_i}\}$) (MLE = Maximum Likelihood Estimation).

---
## Calculate CLTV 
Now we can calculate CLTV with parameters $\alpha_{u_i}$ and $\beta_{u_i}$.
Survival rate at each time point is as follows:  

- $r_{u_i}(t) = (\beta_{u_i} + t - 1) / (\alpha_{u_i} + \beta_{u_i} + t -1)$
- $s_{u_i}(t) = 1 \hspace{5pt} when \hspace{5pt} t = 0$
- $s_{u_i}(t) = r_{u_i}(t)s_{u_i}(t-1) \hspace{5pt} when \hspace{5pt} t > 0$

Then, CLTV for the specific customer $u_i$ is given as follows ($k$ : large enough, integer):
$$
\begin{aligned}
E_{u_i}[CLTV] &= \displaystyle{\sum_{t=0}^{k}} \hspace{2pt}\frac{m}{(1+d)^t}s_{u_i}(t)
\hspace{10pt}...\hspace{10pt}(13)
\end{aligned}
$$
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