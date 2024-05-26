## Stable and bias-free Monte Carlo Greeks <small>using finite differences (FD) and (adjoint)-automatic differentiation (AAD)</small>

In the realm of financial derivatives, accurate and stable computation of sensitivities of option prices to various parameters — is crucial for effective risk management and trading strategies. This becomes particularly challenging when dealing with Monte Carlo simulations, especially for options with discontinuous payoffs (digital, barrier, etc.).

In recent years, automatic differentiation (AAD) has gained significant popularity for computing sensitivities, or Greeks, of financial derivatives. This method promises higher efficiency and accuracy compared to traditional finite difference (FD) approaches. Meanwhile, finite differences are still widely used in many established pricing systems, often employing the bump-and-reval technique to obtain Greeks. However, when applied to Monte Carlo simulations, particularly for options with discontinuous payoffs, both methods have their own sets of challenges and intricacies.

Here, I'll look at various examples that illustrate the problem and also explore different approaches to overcome these issues. As a motivational example, let us directly study the issues of both methods when naively implementing a standard Monte Carlo estimator for simple digital options.

## 0. Introductary Example: Digital call option present value and Delta

Consider the Black-Scholes model allowing for a closed solution of a digital asset or nothing (up-and-in) option. Furthermore, consider a naive Monte Carlo simulation to compute the present value of this option.  

Please check the code given in the [Colab notebook](https://github.com/da-roth/StableAndBiasFreeMonteCarloGreeks/blob/main/src/IntroductoryExample/introductory_example_Colab.ipynb) to reproduce the results of the following images and for the used model and simulation parameters. The notebooks use PyTorch as the AAD framework.

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
- ["Convergence of Milstein Brownian bridge Monte Carlo methods and stable Greeks calculation"](https://arxiv.org/abs/1906.11002) extends our previous work to continuously-monitored barrier options and relaxes model assumptions from Black-Scholes to local volatility models. We address the instability of the well-studied and commonly used Brownian bridge approach for second-order Greeks in barrier options. Our proposed Monte Carlo estimator, which combines the Brownian bridge approach with the concept of  ['Conditioning on one-step survival for barrier option simulations'](https://pubsonline.informs.org/doi/abs/10.1287/opre.49.6.923.10018) by Glasserman and Staum, allows for stable second-order Greeks. To complete the mathematical foundation, we define when a Monte Carlo estimator permits stable second-order Greeks through FD and provide a theorem proving stability under certain conditions.

- ["Automatic backward differentiation for American Monte-Carlo algorithms (conditional expectation)"](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3090009_code373028.pdf?abstractid=3000822&mirid=1&type=2) by Fries. In this work, the author extends automatic differentiation for algorithms containing a conditional expectation operator. In Figure 1, the author observes similar results (as investigated in above's introductary example) for the standard Monte Carlo estimator for a Bermudan digital option:
    <img src="images/FriesBermudan.png" alt="present value comparison" width="400" height="300">

    The blue line demonstrates the bias of the standard Monte Carlo estimator for AAD and the red-dots the instability of FD. For the relation of the FD shift size (plus the amount of Monte Carlo samples) and the degree of instability, please check the descriptions within the above mentioned works. Please check the [homepage](http://christian-fries.de/finmath/stochasticautodiff/) of the author, for further reading on stochastic automatic differentiation.


 Stochastic automatic differentiation allows the user to have reasonable results even if inputing payoffs (e.g. with indicator functions) that would typically lead to biased-results. However, by construction, it is restricted to the specific set classes that were implemented. For example, consider the two-dimensional autocallable case within ["A Monte Carlo pricing algorithm for autocallables that allows for stable differentiation."](https://www.math.uni-frankfurt.de/~harrach/publications/StableDiffs.pdf). Here, we'd have to add the proposed transformation and perhaps a specific node on the graph that would allow the support of these kind of instruments.

All in all, taking all these considerations into account, one topic I'll investigate in the following it the question on how to create new Monte Carlo estimators, that allow for bias-free Greeks using standard AAD.

## 2. Example: Continuation of digital call and barrier options

Instead of using the standard Monte Carlo estimator for digital options, let us use the well-known conditional expectation transformation, see e.g. [Monte Carlo methods in financial engineering](https://link.springer.com/book/10.1007/978-0-387-21617-1) by Glasserman.
Using this improved Monte Carlo estimator, we receive the following results for Delta:

<img src="images/deltaAdvanced.png" alt="present value comparison" width="400" height="300">

The images can be reproduced by executing the code given in this [Colab notebook](https://github.com/da-roth/StableAndBiasFreeMonteCarloGreeks/blob/main/src/ExampleIntrodcutoryContinued/example_continued_Colab.ipynb). It's relatively easy to show that the improved Monte Carlo estimator meets the assumptions of the theorem on stable Greeks by FD formulated by Alm et al., and hence the stable Delta by FD is not surprising. Furthermore, the improved Monte Carlo estimator indeed seems to allow for bias-free AAD.

### 2.1. Example: Barrier options

Let's investigate the behaviour of FD and AAD for an up-and-out continuously-monitored barrier option call option. For the chosen parameters and to re-produce the results please check the [Colab notebook](https://github.com/da-roth/StableAndBiasFreeMonteCarloGreeks/blob/main/src/ExampleBarrier/example_barrier_Colab.ipynb).
While I'll skip the investigation of the present value, the first property I'd like to point out here is as follows. Studying the results for Delta of FD and AAD, we get:

<img src="images/deltaBarrierStandard.png" alt="present value comparison" width="400" height="300">

While again FD leads to instabilities, AAD is biased in such a way that it might even have the wrong sign. The explanation is rather intuitive: While in general an increasing asset value S has positive impact on the 'vanilla' part of the payoff max(S-K,0), the AAD tool doesn't account in that an increasing asset value results in a greater knock-out probability. Hence, path that survived (not crossed the barrier), will always have positive Delta using standard AAD tools. 

For barrier options, the commonly used Brownian-bridge approach, see e.g. [here](https://arxiv.org/abs/1906.11002) and reference therein, leads to the following results for Delta:

<img src="images/deltaBarrierBB.png" alt="present value comparison" width="400" height="300">

However, as investigated in ["Convergence of Milstein Brownian bridge Monte Carlo methods and stable Greeks calculation"](https://arxiv.org/abs/1906.11002), the Brownian-Bridge correction, doesn't allow for stable second-order Greeks. To get second-order Greeks through AAD, it is common to use finite differences on two evaluation of Delta. The following image, demonstrates Gamma through FD and AAD:

<img src="images/gammaBarrierBB.png" alt="present value comparison" width="400" height="300">

Hence, the Brownian-bridge correction, doesn't allow for stable Gamma for barrier options. Furthermore, the AAD approach (finite differences of AAD Deltas) leads to instabilities, too.

If we use the newly propsed Monte Carlo estimator of ["Convergence of Milstein Brownian bridge Monte Carlo methods and stable Greeks calculation"](https://arxiv.org/abs/1906.11002), we get the following results for Gamma:


Hence, the estimator not only allows for stabile Gamma through FD, but also results in stable Gamma through AAD.

<img src="images/gammaBarrierOSS.png" alt="present value comparison" width="400" height="300">

Again, see this [Colab notebook](https://github.com/da-roth/StableAndBiasFreeMonteCarloGreeks/blob/main/src/ExampleBarrier/example_barrier_Colab.ipynb) to reproduce all the results.

Summary: 
1. While the payoff of the barrier option has a discontinuity 

<img src="images/payoffBarrier.png">

given through max(S) < B, the standard Monte Carlo estimator for this payoff would lead to instabilities (FD) and a bias (AAD). 

2. In contrast, a Monte Carlo estimator incorporating the Brownian-bridge correction, leads to stable (FD) and bias-free (AAD) first order Greeks. Furthermore, due to its construction one might also define a pathwise-sensitivities estimator, see e.g. section 7 [here](http://people.maths.ox.ac.uk/~gilesm/files/sylvestre_thesis.pdf). However, the Brownian-bridge corrected Monte Carlo estimator doesn't allow for stable (FD) nor for bias-free (AAD) Greeks, since the incorporated crossing probabilities, given by

<img src="images/bbProbability.png">

lead to a discontinuity within the first derivative, see e.g. (7.7) [here](http://people.maths.ox.ac.uk/~gilesm/files/sylvestre_thesis.pdf). 

3. As seen in above's example, the Monte Carlo estimator proposed in ["Convergence of Milstein Brownian bridge Monte Carlo methods and stable Greeks calculation"](https://arxiv.org/abs/1906.11002), which allows for stable second-order Greeks through FD, also produced bias-free Greeks through AAD. As mentioned in the article, the estimator would also allow the usage of a pathwise sensitivities estimator, since it also got rid of the discontinuity of the first derivative. 


## 3. Connection of stable Greeks through FD and bias-free Greeks through AAD

Above's example for barrier options gives us a good intuition on the requirements of a Monte Carlo estimator to allow for bias-free Greeks through AAD. The intuitive explanation of the connection is as follows: A Monte Carlo estimator allows for bias-free Greeks up to a certain degree, if it is capable of computing these Greeks through a pathwise-sensitivity algorithm. The studies in ["Convergence of Milstein Brownian bridge Monte Carlo methods and stable Greeks calculation"](https://arxiv.org/abs/1906.11002) indicate that this is the case if the Monte Carlo estimator allows for stable Greeks through FD.

While the creation of the pathwise sensitivities calculator for these options is rather time-consuming, one might prefer to create the Monte Carlo estimator in such a way that it allows for bias-free AAD and then use an AAD framework instead of implementing the pathwise-sensitivity algorithm, in practice. 

If a stocastic AAD framework should handle the standard Monte Carlo estimator as an input and still be able to produce stable second-order Greeks, it would require to automatically handle the two arising discontinuities - in such a way as the Brownian-bridge correction and the one-step survival correction do. Also being time-consuming, it might be usable for other types of options, such as other types of barrier options (knock-in, knock-down-in/out etc.), as well e.g. for Bermudan American options, as studies in one of the articles by [Fries](http://christian-fries.de/finmath/stochasticautodiff/) and hence having some nice potential.

## 4. Bias-Free Stable (BFS) Monte Carlo estimators for financial instruments

In this section, we will take an in-depth look at deriving Monte Carlo estimators from the perspective of various financial instruments. I'll denote Monte Carlo estimators that allows for stable FD and bias-free AAD Greeks (at least up to second-order) by Bias-Free Stable (BFS) Monte Carlo estimators.

Before jumping into the examples, let me refer to the monograph ["Quantitative Finance: Back to Basic Principles"](https://books.google.de/books?hl=en&lr=&id=rLsxBgAAQBAJ&oi=fnd&pg=PP1&ots=1wmraZ3t1W&sig=K-irQaky7v9VS-5QWLFuj5EjROQ&redir_esc=y#v=onepage&q&f=false) by Adil Reghai. In chapter 3 of this book, the author describes (coming from a PnL point of view) the validity of the Black & Scholes model for different instruments. While the chapter concludes that for some instruments (e.g. European Call options) the Black & Scholes is appropriate, I'll nevertheless try consider the payoff structure of these instruments in the following. There is a simple reason for this: even more complex instruments, for which a Monte Carlo estimator might be required, often have features of standard instruments.

1. European options: Even for the simple European Call option, we see that the naive Monte Carlo estimator results in infeasible Gamma somputation. The reason is again quite simple: The derivative of the maximum function contains an indicator function. In this case, intuitively speaking, the BFS Monte Carlo estimator can be derived by forcing the path to end above the strike price (and a proper normalization). For a European Call option, check out this [Colab notebook](https://github.com/da-roth/StableAndBiasFreeMonteCarloGreeks/blob/main/src/BFS_Examples/example_Europ_Call_Colab.ipynb).

2. Digital options: Discussed in the introductory example. [Colab notebook](https://github.com/da-roth/StableAndBiasFreeMonteCarloGreeks/blob/main/src/ExampleIntrodcutoryContinued/example_continued_Colab.ipynb)

3. Barrier options: Discussed in section 2. [Colab notebook](https://github.com/da-roth/StableAndBiasFreeMonteCarloGreeks/blob/main/src/ExampleBarrier/example_barrier_Colab.ipynb)






