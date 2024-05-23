## Stable and bias-free Monte Carlo Greeks <small>using finite differences (FD) and (adjoint)-automatic differentiation (AAD)</small>

In the realm of financial derivatives, accurate and stable computation of sensitivities of option prices to various parameters — is crucial for effective risk management and trading strategies. This becomes particularly challenging when dealing with Monte Carlo simulations, especially for options with discontinuous payoffs (digital, barrier, etc.).

In recent years, automatic differentiation (AAD) has gained significant popularity for computing sensitivities, or Greeks, of financial derivatives. This method promises higher efficiency and accuracy compared to traditional finite difference (FD) approaches. Meanwhile, finite differences are still widely used in many established pricing systems, often employing the bump-and-reval technique to obtain Greeks. However, when applied to Monte Carlo simulations, particularly for options with discontinuous payoffs, both methods have their own sets of challenges and intricacies.

In this blog/repo, I want to look at various examples that illustrate the problem and also explore different approaches to overcome these issues. As a motivational example, let us directly study the issues of both methods when naively implementing a standard Monte Carlo estimator for simple digital options.

## 0. Introductary Example: Digital call option present value and Delta

Consider the Black-Scholes model allowing for a closed solution of a digital asset or nothing (up-and-in) option. Furthermore, consider a naive Monte Carlo simulation to compute the present value of this option and the following model and simulation parameters.

    | Parameter | Description       | Value    |
    |-----------|-------------------|----------|
    | t_0       | initial time      | 0.0      |
    | T         | time to maturity  | 1.0      |
    | r         | risk-free rate    | 0.04     |
    | q         | dividend  rate    | 0.0     |
    | sigma     | volatility        | 0.3      |
    | K         | strike price      | 50       |

    | N         | MC samples        | 100k     |
    | h         | finite dif. para. | 0.0001   |


Please check the [Colab notebook](https://github.com/da-roth/StableAndBiasFreeMonteCarloGreeks/blob/main/src/IntroductoryExample/introductory_example_Colab.ipynb) to reproduce the results of the following images.

### Present value comparison

 First, let us compare the present value obtained via the standard naive Monte Carlo estimator (100k samples) and the closed-form solution:

 <img src="images/presentValueClosedAndNaive.png" alt="present value comparison" width="400" height="300">

 Normally, one might want to study convergence, but for the purpose of this investigations, I'll here simply perceive that the standard Monte Carlo estimator produces viable results for different initial spot values S.


### Delta comparison

 In the next figure, we'll take a look at the Deltas computed through FD and AAD and compare them to the exact result: 

 <img src="images/deltaClosedAndNaive.png" alt="Delta comparison" width="400" height="300">

 While the naive Monte Carlo approach shows the typical instabilities of Monte Carlo simulation for discontinuous payoff, the AAD applied to the naive approach does not suffer from instability, its results are biased.

## 1. Introduction

Let me highlight the key contributions from the literature that have shaped my investigations into stable and bias-free Greeks calculations using Monte Carlo methods. These articles cover foundational concepts, innovative techniques, and advancements in the stability of differentiation and sensitivity calculations for financial derivatives. While each paper contains a broad array of content, the following key points are especially pertinent to the focus of this page:


- ["A Monte Carlo pricing algorithm for autocallables that allows for stable differentiation."](https://www.math.uni-frankfurt.de/~harrach/publications/StableDiffs.pdf) by Alm et al.. In this work, section 2.3 formulates a definition of when a Monte Carlo estimator allows for stable differentiation through finite differences (FD). Additionally, the authors provide a theorem proving stability under specific assumptions.
- ["Monte Carlo pathwise sensitivities for barrier options"](https://www.risk.net/journal-of-computational-finance/7533966/monte-carlo-pathwise-sensitivities-for-barrier-options) by my co-authors and me, introduces a transformation of the barrier option's payoff, enabling stable finite differences. We developed a framework that computes pathwise sensitivities successively, effectively replacing finite differences. [Algorithm 1](https://www.math.uni-frankfurt.de/~harrach/publications/pathwise.pdf) on page 9 of the article's preprint clearly illustrates the connection to ['forward accumulation'](https://en.wikipedia.org/wiki/Automatic_differentiation#Forward_and_reverse_accumulation) in autmatic differentiation.
- ['Convergence of Milstein Brownian bridge Monte Carlo methods and stable Greeks calculation'](https://arxiv.org/abs/1906.11002) extends our previous work to continuously-monitored barrier options and relaxes model assumptions from Black-Scholes to local volatility models. We address the instability of the well-studied and commonly used Brownian bridge approach for second-order Greeks in barrier options. Our proposed Monte Carlo estimator, which combines the Brownian bridge approach with the concept of  ['Conditioning on one-step survival for barrier option simulations'](https://pubsonline.informs.org/doi/abs/10.1287/opre.49.6.923.10018) by Glasserman and Staum, allows for stable second-order Greeks. To complete the mathematical foundation, we define when a Monte Carlo estimator permits stable second-order Greeks through FD and provide a theorem proving stability under certain conditions.

- ['Automatic backward differentiation for American Monte-Carlo algorithms (conditional expectation)'](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3090009_code373028.pdf?abstractid=3000822&mirid=1&type=2) by Fries. In this work, the author extends automatic differentiation for algorithms containing a conditional expectation operator. In Figure 1, the author observes similar results (as investigated in above's introductary example) for the standard Monte Carlo estimator for a Bermudan digital option:
    <img src="images/FriesBermudan.png" alt="present value comparison" width="400" height="300">

    The blue line demonstrates the bias of the standard Monte Carlo estimator for AAD and the red-dots the instability of FD. For the relation of the FD shift size (plus the amount of Monte Carlo samples) and the degree of instability, please check the descriptions within the above mentioned works. Please check the [homepage](http://christian-fries.de/finmath/stochasticautodiff/) of the author, for further reading on stochastic automatic differentiation.


 Stochastic automatic differentiation allows the user to have reasonable results even if inputing payoffs (e.g. with indicator functions) that would typically lead to biased-results. However,by construction, it is restricted to the specific set classes that were implemented. For example, consider the two-dimensional autocallable case within ["A Monte Carlo pricing algorithm for autocallables that allows for stable differentiation."](https://www.math.uni-frankfurt.de/~harrach/publications/StableDiffs.pdf). Here, we'd have to add the proposed transformation and perhaps a specific node on the graph that would allow the support of these kind of instruments.

All in all, taking all these considerations into account, one topic I'll investigate in the following it the question on how to create new Monte Carlo estimators, that allow for bias-free Greeks using standard AAD.

## 2. Example: (Continuation of Introductary Example)

Instead of using the standard Monte Carlo estimator for digital options, let us use the well-known conditional expectation transformation, see e.g. [Monte Carlo methods in financial engineering](https://link.springer.com/book/10.1007/978-0-387-21617-1) by Glasserman.
Using this improved Monte Carlo estimator, we receive the following results for Delta:

<img src="images/deltaAdvanced.png" alt="present value comparison" width="400" height="300">

The images can be reproduced by executing the code given in this [Colab notebook](https://github.com/da-roth/StableAndBiasFreeMonteCarloGreeks/blob/main/src/ExampleIntrodcutoryContinued/example_continued_Colab.ipynb). It's relatively easy to show that the improved Monte Carlo estimator meets the assumptions of the theorem on stable Greeks by FD formulated by Alm et al., and hence the stable Delta by FD is not surprising. Furthermore, the improved Monte Carlo estimator indeed seems to allow for bias-free AAD.



