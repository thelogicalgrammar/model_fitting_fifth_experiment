#### Hierarchical Bayesian model

The data will be analyzed with a Bayesian hierarchical model coded with PyMC3, a Python implementation of Hamiltonian Monte Carlo algorithms and related tools for Bayesian statistics. The response variable is the number of times the participant judged that the presented stimulus belongs to the category. The predictor variables are the condition, the stimuli presented as belonging to the category that the participant should infer, and how many times the participant is asked to categorize the stimuli.

There are three levels in the hierarchical model. The bottom level encodes how many times a specific participant judged a specific stimulus as belonging or not to the category. The middle level clusters the single judgments into participants. The top level encodes information about clusters of participants. 

At the bottom level, the participants’ judgments for the stimuli are distributed Binomially, and they are coded as the number of times the participant judged that the presented stimulus belongs to the category. The Binomial $n$ parameter is the number of times that the participant was asked to categorize that stimulus. The $p$ parameter for stimulus $i$ and participant $j$ is the sum of the probabilities that $j$ attributes to all the categories that contain $i$ (*hypothesis averaging* from Xu & Tenenbaum, 2007), plus an error applied that is estimated for $j$. The categories are the convex set of stimuli, i.e. any set of stimuli such that if two stimuli $s$ and $p$ belong to the set, then any stimulus $q$ between $s$ and $p$ also belongs to the set.

The posterior probability over categories for each participant is calculated at the second level of the hierarchical model. For agent $j$, it is a function of three things: which stimuli are presented to $j$ as belonging to the category, $j$’s prior preference for monotonic categories, and $j$’s prior preference for categories of certain sizes. The posterior of $j$ for a category $k$ is proportional to the product of the likelihood, i.e. the probability of the presented stimuli $v$ given $k$, and the probability that $j$ attributes a priori to $k$. The probability $p(v | k)$ is calculated following the size principle discussed in Xu & Tenenbaum (2007) (see the full specification of the model below).

The population level parameters are the following. (1) Population level distribution(s) of preferences for monotonic hypotheses. This is two distributions, one per condition. (2) A population level distribution of preference for large or small hypotheses. Same for both conditions. (3) A population level distribution of production error. Same for both conditions. 

The hyperpriors can be read below from the full description of the model.


Full model:

\begin{align}
\textrm{ACCEPT}_{i, j} & \sim \textrm{Binomial}(\omega_{j, i}, \phi_{j, i}) \\
\\
\phi_{j, i} & = g(\textrm{PE}_j, \sum_{h \in \mathbb{H}}p_j(h) \mathbb{1}_{i \in h})\\
p_j(h) & = p(h | \textrm{PL}_j, \textrm{PM}_j, \textrm{OBS}_j) \quad \textrm{for } h \in \mathbb{H}\\
p(h | \textrm{PL}_j, \textrm{PM}_j, \textrm{OBS}_j) & \propto p(\textrm{OBS}_j | h, \textrm{PL}_j, \textrm{PM}_j) p(h | \textrm{PM}_j, \textrm{PL}_j) = p(\textrm{OBS}_j | h) p(h | \textrm{PM}_j, \textrm{PL}_j)  \\
p(\textrm{OBS}_j | h) & = \mathbb{1}_{\textrm{OBS}_j \subseteq h}\frac{1}{|h|^{|\textrm{OBS}_j|}} \\
p(h | \textrm{PM}_j, \textrm{PL}_j) & \propto p(h | \textrm{PM}_j) p(h | \textrm{PL}_j) \\
p(h | \textrm{PM}_j) & = \textrm{PM}_{j} \mathbb{1}_{\textrm{mon}}(k) + (1 – \textrm{PM}_{j})(1 – \mathbb{1}_{\textrm{mon}}(k)) \\
p(h | \textrm{PL}_j) & = e^{|h| \textrm{PL}_j} \\
\\
\textrm{PL}_j & \sim \mu_{\textrm{PL}} + \sigma_{\textrm{PL}} \textrm{Normal}(0, 1) \\
\textrm{logit}(\textrm{PM}_j) & \sim \mu_{\textrm{PM, CONDITION}_j} + \sigma_{\textrm{PM, CONDITION}_j} \textrm{Normal}(0, 1)\\
\textrm{PE}_j & \sim \textrm{HalfNormal}(\sigma_{\textrm{PE}})\\
\\
\mu_{\textrm{PL}} & \sim \textrm{Normal}(0, 2) \\
\sigma_{\textrm{PL}} & \sim \textrm{HalfNormal}(2) \\
\mu_{\textrm{PM, CONDITION}_j} & \sim \textrm{Normal}(0, 2) \\
\sigma_{\textrm{PM, CONDITION}_j} & \sim \textrm{HalfNormal}(1) \\
\sigma_{\textrm{PE}} & \sim \Gamma(1, 2)\\
\end{align}

Where 
- $\textrm{ACCEPT}_{i, j}$ is the number of times participant $j$ accepted stimulus $i$ as belonging to the category.
- $\omega_{j, i}$ is the number of times participant $j$ was asked to categorize stimulus $i$.
- $\phi_{j, i}$ is the probability of participant $j$ accepting stimulus $i$.
- $\mathbb{H}$ is the set of convex hypotheses.
- $h$ is used ambiguously as a set of stimuli, e.g. in $i\in h$, and the event that the hypothesis to infer is the set $h$, e.g. $p(h)$. Context should disambiguate how each instance is meant.
- $\mathbb{1}_{i \in h}$ is the indicator function for whether hypothesis $h$ contains stimulus $i$.
- $g(x, y) = y + (\frac{1}{1 + e^{-x}}-0.5)(1-2y)$. For all values of $y$, $g(x, y) \to 0.5$ as $x\to\inf$ and $g(x, y) = y$ if $x=0$. We use this function to model the error in $j$'s production behaviour. As the error $\to \inf$, the choice of whether to accept or not a stimulus becomes uncorrelated with the participant's biases.
- $\textrm{PE}_j$ is $j$'s production error.
- $\textrm{PL}_j$ is $j$'s preference for large hypotheses.
- $\textrm{PM}_j$ is $j$'s preference for monotonic hypotheses.
- $\textrm{OBS}_j$ is the set of stimuli that were presented to $j$ as belonging to the category.
- $\textrm{CONDITION}_j$ is the condition of participant $j$, coded as 0 (for distance condition) or 1 (for property condition).
- We parameterize the $\Gamma$ distribution with $\alpha$ and $\beta$.

The simpler model assumes that the participant's preferences for monotonicity are sampled from a single distribution, rather than difference distributions for the two conditions.

#### Hypothesis testing

Call $x^{s}$ the value of model variable $x$ at the $s^{th}$ posterior sample, out of $S$ samples. We calculate the posterior distribution of differences between the means of the population-level distribution over preferences for monotonicity between two conditions $ \{ \mu^{s}_{\textrm{PM}, \textrm{CONDITION}_0} - \mu^{s}_{\textrm{PM}, \textrm{CONDITION}_1} \}_{i=1}^S$. We calculate the 95\% HPD interval, and we accept the hypothesis that there is a difference between the two conditions with respect to the preference for monotonic hypotheses iff the HPD interval does not contain 0.

#### Practicalities

This document contains the code to fit the model to the experimental data and analyze the results. Fitting the model is computationally intensive, and the LOO CV requires numerous fits. Therefore, we add code to run the model on the server. The code is written for Oracle Grid Engine (https://en.wikipedia.org/wiki/Oracle_Grid_Engine), but other implementations can be written based on the code below.

#### References

Xu, Fei, and Joshua B. Tenenbaum. 2007. “Word Learning as Bayesian Inference.” Psychological Review 114 (2): 245–72. https://doi.org/10.1037/0033-295X.114.2.245.