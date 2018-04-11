*(This article is translated from Mr. Zheng Huabin‘s 《令人拍案叫绝的Wasserstein GAN》originally posted on Zhihu. The translator has already received Mr. Zheng's authorization for the publication of the translated version. For native speakers / those who are interested in reading the Chinese version, [here's the link.](https://zhuanlan.zhihu.com/p/25071913))*

While the researching GAN is becoming so popular that doing them starts to become somewhat cliché, a new paper posted on arXiv called [Wasserstein GAN](https://arxiv.org/abs/1701.07875) has sparked a huge discussion on the Machine Learning subreddit, to a point where even [Ian Goodfellow himself joined the talk](https://www.reddit.com/r/MachineLearning/comments/5qxoaz/r_170107875_wasserstein_gan/). So, what’s so unique about this new theory?

Just FYI, since their first appearance in 2014, GAN’s implementations have suffered from training difficulties, loss of generative/discriminative (G/D) models being unable to identify the training process, as well as collapsing generated samples. Scholars did attempt to improve, yet most of the papers made little improvement, like the enumerative methodology introduced by the [DCGAN](https://arxiv.org/abs/1511.06434) paper, where searches are done on different generative/discriminative model architectures. But this doesn’t help the big picture, except that Wasserstein GAN (WGAN) did. In general, WGAN exhibits the following mind-blowing characteristics:

*   It completely solves the instability of GAN training, where concerns on balance between generative/discriminative model training are no longer needed;
*    It generally solves the collapse mode problem and ensures the diversity of the generated samples;
*    A metric finally exists (just like accuracy/cross entropy) to indicate the process of training, where a smaller value indicates a better generative model;
*    The above can be achieved without a carefully designed model architecture; even a naïve MLP is capable of converging.

The snapshot of the algorithm can be found below:

![Imgur](https://i.imgur.com/y7U95UI.jpg)

But where do these benefits come from? This is the part where the Wasserstein GAN is truly “Wonderful” (hence the alliteration). It takes the original author of WGAN two entire papers to explain everything: the theorems in [“Towards Principled Methods for Training Generative Adversarial Networks”](https://arxiv.org/abs/1701.04862) that provides an analysis to why the original GAN doesn’t function and thus the ways to improve it, and the following “Wasserstein GAN” that provides an additional set of theorems that finalizes this novel methodology. **Intriguingly, the new algorithm only made four alterations to the original GAN:**
1.    Removing the sigmoid activation function in the last layer
2.    Removing the log in the loss function of G/D models
3.    Clipping the parameters so that their absolute values are smaller than a constant c
4.    Not using any optimization methods based on momentum (like the naïve method and Adam); RMSProp is the recommended method, though SGD works as well

The simplicity of these modifications makes it so deceivingly astonishing that people on Reddit questions: *is that it? Isn’t there any other hidden details?* This reminds me of another cliché story on the net, which says that an experienced engineer can earn ten grand by simply drawing the correct line on the shell of the motor that identifies the annoying problem. Drawing a line worths 1 dollar, while knowing where to draw it gives you the rest 9999. The difference is pretty evident, and the four alterations are the four lines by our author Martin Arjovsky. At this point, it is already enough for implementation from an engineering perspective, but the “knowing where to draw the line” part roots deeply from sophisticated mathematical analyses, which is the primary goal of this essay.

**The following paragraphs are organized into the following five sections:**
1.    What’s really not working in the original GAN *(this part is pretty long)*
2.    A remedial solution before the advent of WGAN
3.    The superior characteristics exhibited by the Wasserstein distance
4.    From Wasserstein distance to WGAN
5.    Conclusion

*Understanding the proofs and theorems in the original paper requires rather extensive knowledge on **measure theory** and **topology**, so instead this paper will focus on a more **intuitive** aspect, where the interpretations are based off lower-dimensional examples to help readers to ponder the ideas behind the scene while maintaining mathematical rigor. You are welcomed to point out any errors in the comment section.*

*From now on, “Wasserstein GAN” is referred as “The original WGAN,” while “Towards Principled Methods for Training Generative Adversarial Networks” is referred as “Preliminaries on WGAN.”* 

**Section1: What’s really not working in the original GAN**

To recap, original GAN paper proposes that the goal of the discriminator is to minimize the following loss function, where a sample from the model distribution has a positive effect and vice versa.

$$-\mathbb{E}_{x \sim P_r}[\log D(x)]-\mathbb{E}_{x \sim P_g}[\log (1-D(x))]$$
*(Equation 1)* 

 In the equation, $$P_{r}$$ refers to the model distribution, and $$P_{g}$$ refers to the implicitly defined generative network distribution. For the generative network, Goodfellow initially presented a loss function, to which a refined version was also proposed. One can found both of them respectively below:

$$\mathbb{E}_{x \sim P_{g}}[\log (1-D(x))]$$
*(Equation 2)*
$$\mathbb{E}_{x \sim P_{g}}[-\log D(x)]$$
*(Equation 3)*  
 
The latter is referred as “the – log D alternative” or “the – log D trick” in the two WGAN papers. The original WGAN analyzed respectively the problems with these two loss functions, both of which will be explained below:

_**The problem with the first form of the original GAN**_

**Tl; DR: The better the discriminator is, the worse the vanishing gradient effect will be.**

**The long version:** The “Preliminaries on WGAN” presented two perspectives on this problem, the first of which starts with taking a closer look at the equivalent form of the generative network loss function.

Firstly, from *Equation 1* we can obtain the optimal discriminator by looking at an arbitrary sample $$x$$. Since it can come from either the model distribution or the generative distribution, its contribution to the overall loss can be expressed as 

$$-P_{r}(x)\log D(x) - -P_{g}(x)\log [1-D(x)]$$
 
Solve by setting the derivative of $$D(x)$$ to 0, we get

$$-\frac{P_{r}(x)}{D(x)}+\frac{P_{g}(x)}{1- D(x)}=0$$
 
Furthermore, by simplifying the expression, we can finally get the optimal discriminator as

$$D^*(x)=\frac{P_{r}(x)}{P_{r}(x)+P_{g}(x)}$$
*(Equation 4)*  
 
The result is straightforward to interpret intuitively since it only deals with the relative ratio between the likelihood of the sample coming from the model distribution and the generative distribution. If $$P_r(x)=0$$ and $$P_g(x) \neq 0$$, the optimal discriminator should confidently produce the output 0; if $$P_r(x)=P_g(x)$$, which means that there is a half-half chance of sample being true/false, the optimal discriminator should also express this indecisiveness as an output of 0.5.

But GAN has a trick not to train the discriminator too well, or the generative network will stop converging (its loss stop decreasing). To understand why this is the case, we take a look at the extreme condition: what the loss function of the generative network would look like if the optimal discriminator is obtained. Adding a term that does not depend on the generative network to *Equation 2*, we obtain
 
 $$\mathbb{E}_{x \sim P_{r}}[\log D(x)] + \mathbb{E}_{x \sim P_{g}}[\log (1-D(x))]$$
 
It is important to notice that this is exactly the additive reciprocal of loss function of the discriminator. By plugging in *Equation 4*, some simplifications will result in the following:

 $$\mathbb{E}_{x \sim P_{r}} \log \frac{P_r(x)}{\frac{1}{2}[P_r(x)+P_g(x)]} + \mathbb{E}_{x \sim P_{g}} \log \frac{P_g(x)}{\frac{1}{2}[P_r(x)+P_g(x)]}  - 2 \log 2$$
  *(Equation 5)*
    
The presentation of this seemingly weird form is to help introduce the **Kullback-Leibler divergence** (KL divergence) and the **Jensen-Shannon divergence** (JS divergence), two important metrics in the original GAN framework that are the primary targets of criticism in the WGAN papers (where they are replaced by the Wasserstein distance). The exact definition of these two metrics can be found below:

 $$KL(P_1||P_2)=\mathbb{E}_{x \sim P_{1}} \log \frac{P_1}{P_2}$$
  *(Equation 6)*
  
 $$JS(P_1||P_2)=\frac{1}{2}KL(P_1||\frac{P_1+P_2}{2})+\frac{1}{2}KL(P_2||\frac{P_1+P_2}{2})$$
  *(Equation 7)*
 
Thus, Equation 5 can be rewritten as 
 
 $$2JS(P_r||P_g)-2\log2$$
  *(Equation 8)*
  
The readers are welcomed to take a deep breath at this point and see the conclusions we have made so far:  **According to the generative network loss defined in the original GAN, we can obtain an optimal discriminator, and thus rewrite the generative loss as the JS divergence between the model distribution $P_r$ and the generative distribution $P_g$. The goal of training the discriminator in per se is to minimize this JS divergence.**

That’s where the problem comes in. Of course, we would like to minimize the JS divergence between two distributions, where $$P_g$$ is being “pulled” towards $$P_r$$ and eventually can forge the examples. The above is true when the two distribution intersects, but what happens when the two distributions do not overlap, or their intersection is trivially small that it is safe to ignore them (a concept that will be explained below)? What will the JS divergence be like?
The answer is $$\log2$$ since there are only four possible cases for an arbitrary x:
 
 $$P_1(x) = 0 \space \text{and} \space P_2(x) = 0$$
 $$P_1(x) \neq 0 \space \text{and} \space P_2(x) \neq 0$$
 $$P_1(x) = 0  \space \text{and} \space  P_2(x) \neq 0$$
 $$P_1(x) \neq 0 \space \text{and} \space P_2(x) = 0$$
 
The first one totally does not contribute to the JS divergence, so does the second due to the negligibility of the intersection. According to the definition of JS divergence, this third case evaluates to:
$$\log \frac{P_2}{\frac{1}{2}(P_2+0)}= \log2$$

Since the fourth case is also analogous, in the end

$$JS(P_r||P_g)=\log2$$
 
In other words, no matter how far they are apart from each other, as long as the distributions are not “touching each other with a considerably large intersection,” the JS divergence is fixed to the constant $$\log2$$, which gives a gradient 0 for the gradient descent method. This is terrible news cause the generative network is unable to get any information from the optimal discriminator; and for the almost-optimal discriminators, the threat of vanishing gradients still exists.

But how likely is it that the distributions $$P_r$$ and $$P_g$$ do not share a cross-section that is large enough? The TL;DR version again is very, very likely. The more rigorous version is: **If the supports of the two distributions are low-dimensional manifolds in the high-dimensional space, the possibility that the measure of the intersection between them is zero is one.**

Don’t be scared of the jargons that just jumped right at you, and don’t close the webpage yet. Although what the paper gave us were those really rigorous mathematical expressions, intuitively it is easier to comprehend. First, let us take a look at these concepts mentioned above:

* Support: it’s simply the non-negative subset of some function. The support of *Rectified Linear Unit function* (ReLU) is simply $$(0,+\infty)$$, and the support of a probability distribution is the set of all states where the probability density is non-zero.
* Manifold: It is the extension of the curve and surface concept in high-dimensional space. To comprehend manifold in a low-dimensional sense, a surface in the three-dimensional space is considered as a two-dimensional manifold, since its intrinsic dimension is only 2. An arbitrary point on this surface only has a degree-of-freedom of 2. Similarly, a line in three/two-dimensional space is just a one-dimensional manifold.
* Measure: It is the extension of the length/area/volume concept in high dimensional space. One can simply think of them as a measure of “hyper-volume.”

Now, let’s return to the statement above. It is very likely that “the supports of the two distributions are low-dimensional manifolds in the high-dimensional space,” the reason of which roots from the process of the generative network. To produce a sample, the generative requires input from a pre-defined noise prior (usually multivariate Gaussian), which is most of the time low-dimensional. After such a low-dimensional “seed” (like 100-D) is passed into the forward computational graph, a high dimensional picture is produced (like the dimension of a 64*64 picture is 4096), multiple outputs of which also implicitly defines a high-dimensional generative distribution. If the parameters of the generative network are fixed, though the distribution is defined in a high-dimensional space (like 4096), the variation of which is merely restricted by the low-dimensional support (like the 100-D “seed”). Its intrinsic dimension is much, much lower. Counting in the dimensionality-reduction effect of a neural-network mapping, the intrinsic dimension of the example mentioned above can be even smaller than 100. Thus, the support of this 4096-D generative distribution can only be a low-dimensional manifold whose dimension can be no more than 100, which makes the distribution “unable to extend” to the entire space.

The adverse effect of being “unable to extend” is that the model distribution is very unlikely to “collide” with an arbitrarily defined generative distribution. This is easy to comprehend in two-dimensional space. On the one hand, if one picks two random curves in a two-dimensional plane, the probability that they have overlapping section is 0. On the other hand, although they are very likely to have intersecting points since a point is one dimension lower than a line, its measure is 0, and that makes it trivial to the question. The same thing works similarly in three-dimensional space, where though two random surfaces may share the same lines, their measure is 0 because a line is one dimension lower than a surface. Thus we can arrive at another conclusion: if the generative network is initialized randomly, $$P_g$$ is highly unlikely to have any relation to $$P_r$$, and their supports either do not intersect at all, or the intersections has a dimension that is lower than the lowest dimension in $$P_r$$ and $$P_g$$, hence having a measure of 0. So the “measure of the intersection between them is zero” is just a fancy way of saying “not intersecting/ intersections are too small to be significant.”

Then we can obtain the first argument on the vanishing gradient in the generative network of the original GAN: **given an (approximately) optimal discriminator, minimizing the loss of the generative network is equivalent to minimizing the JS divergence between Pr and Pg, and since they are highly unlikely to have any intersections, no matter how far the two distributions are apart from each other, their JS divergence is always $$\log2$$, which results in (approximately) zero gradient.**

Then the preliminary GAN analyzes the problem again in the second perspective with tons of formulae and theorems, but the idea behind them is pretty intuitive:
*    First of all, it is almost impossible for $$P_r$$ and $$P_g$$ to have a nonnegligible intersection, so no matter how close they are from each other, there exists an optimal hyperplane to separate most of them, maybe expect the trivial points where they do intersect.
*    As a universal approximator, an artificial neural network can approximate this hyperplane to infinite precision, so an optimal discriminator do exist that give almost all model examples 1 and all generative examples 0. This discriminator does struggle on the samples that reside at the intersection of the two distributions, but since the set has a measure of 0, it is okay to ignore their effects.
*    The normalized possibilities that the optimal discriminator provide on both the model distribution and the generative distribution are all constant (1/0), so the gradient of the generative network loss is always 0, hence the vanishing gradient.

With these theoretical analyses, we finally know why the original GAN is so unstable: Discriminator too good? Vanishing gradients that completely stops the convergence of generative network loss. Discriminator too bad? The generative network cannot receive a good clue on the update direction, the gradient wiggles. The only time it works is when you train a discriminator that is neither too good or too bad, and it is tough to figure out how to do so.

Here's the empirical result, obtained from the "Preliminaries on WGAN":

![Imgur](https://i.imgur.com/0F0mbw6.png)
*Pay attention to the log scale on the y-axis*

_**The problem with the second form of the original GAN**_

**Tl; DR: The equivalent form of the second loss function is an unreasonable distance measure that not only destabilizes the gradient but also causes the mode to collapse (not enough diversity).**

**The long version:** The "Preliminaries on WGAN" also analyzed this form by two different perspectives. However, I’m only able to find an intuitive explanation for the first perspective. So, if you are interested in the second one, I suggest [reading the original paper](https://arxiv.org/abs/1701.04862).

As mentioned in the previous paragraphs, Ian Goodfellow’s proposed “- log D trick” changed the generative network loss into
 
 $$\mathbb{E}_{x \sim P_{g}}[-\log D(x)]$$
*(Equation 3)*  


And we’ve already known what an optimal discriminator looks like:
$$\mathbb{E}_{x \sim P_{r}}[\log D^*(x)]+\mathbb{E}_{x \sim P_{g}}[\log (1-D^*(x))] = 2JS(P_r||P_g)-2\log2$$
*(Equation 9)*  

 
Then, it is possible to rewrite the KL divergence of $$P_g$$ compared to $$P_r$$ into an expression that contains $$D^*$$ (the optimal discriminator):
 
 $$\begin{split} KL(P_g||P_r) & =\mathbb{E}_{x \sim P_{g}}[\log\frac{ P_{g}(x)}{ P_{r}(x)}] \\ & = \mathbb{E}_{x \sim P_{g}}[\log\frac{ P_{g}(x)/(P_{r}(x)+P_{g}(x))}{ P_{r}(x)/(P_{r}(x)+P_{g}(x))}]  \\ & =\mathbb{E}_{x \sim P_{g}}[ \log \frac{1-D^*(x)}{D^*(x)}] \\ & = \mathbb{E}_{x \sim P_{g}}\log[  1-D^*(x)] - \mathbb{E}_{x \sim P_{g}}\log D^*(x)\end{split}$$
*(Equation 10)* 

Finally, according to *Equation 3,9, and 10*, we can rewrite this trick regarding KL and JS divergences between the two distributions:
 
 $$\begin{split}  \mathbb{E}_{x \sim P_{g}}[-D^*(x)] & = KL(P_g||P_r) -  \mathbb{E}_{x \sim P_{g}}\log[ 1-D^*(x)] \\ & = KL(P_g||P_r) - 2JS(P_r||P_g)+2\log2+\mathbb{E}_{x \sim P_{r}} \log D^*(x)\end{split}$$
 
Since the last two terms do not depend on the generative network G, so minimizing equation 3 is equivalent to minimizing
 
 $$KL(P_g||P_r)-2JS(P_r||P_g)$$
 *(Equation 11)* 
 
Which is really problematic. Intuitively, you are pulling $$P_r$$ towards $$P_g$$ while pushing $$P_g$$ away from $$P_g$$ two times harder, which totally doesn’t make sense. This ridicule is then (of course) demonstrated on the instability of gradient updates.

Taking one step back, even the seemingly normal KL divergence term is, in fact, problematic as well. Recall that since KL divergence is asymmetric, $$KL(P_g||P_r)$$ means something different from $$KL(P_r||P_g)$$ (Ian Goodfellow provides a perfect example in his book [Deep Learning](http://www.deeplearningbook.org/), and I strongly recommend readers to check it out after finishing the article). Take the former for example:

*   When $$P_g(x)$$ approaches 0 and $$P_r(x)$$ approaches 1, $$P_g(x)\log\frac{P_g(x)}{P_r(x)} \rightarrow 0$$; no contributions whatsoever to the KL divergence
*    When $$P_g(x)$$ approaches 1 and $$P_r(x)$$ approaches 0, $$P_g(x)\log\frac{P_g(x)}{P_r(x)} \rightarrow +\infty$$; massive contributions to the KL divergence

In other words, $$KL(P_g||P_r)$$ is biased on how to punish these two errors produced above. The first one in layman’s words is “Generator cannot generate a real example,” which has almost no punishment on the network; The second one, however, that “generator generates a false example,” is being punished significantly by the network. **This discrepancy forced the generative network to produce repetitive but rather “safe” samples instead of a wider spectrum of diverse samples because the latter triggers the second punishment. This effect is often referred to as “collapse mode.” **

**Summary on part 1: When the original GAN obtains an (approximately) optimal discriminator, the first generator loss suffers from vanishing gradient while the second loss faces the menace of absurd optimization goal, unstable gradient, and mode collapse caused by biased punishments.**

here's the image provided also by "Preliminaries on WGAN"
![Imgur](https://i.imgur.com/4Ry4zZm.png)

**Part 2: A remedial solution before the advent of WGAN**

Again, the problem with the original GAN is twofold: the unreasonable optimization goal produced by KL/JS divergence metrics, and the difficulty for an arbitrary distribution to overlap with the model distribution.

The preliminaries of WGAN proposed a solution to the second problem, which is adding noise to the generative and discriminative examples. Intuitively, this causes the two low-dimensional manifolds to “diffuse” to the entire high-dimensional space, forcing a significant overlap. And once that overlap occurs, the JS divergence can truly shine and start to bring those two distributions together because instead of a constant, the JS divergence will decrease as more diffused fragments begin to create overlaps. This in a way solves the vanishing gradient problem exhibited by the first form of GAN. Additionally, as training progresses, we can decrease the variance of the noise by the annealing process, even removing the noise as the original low dimensional manifolds come into contact.  The JS divergence can continue its functionality, producing meaningful gradients that help to bring the two manifolds together. This is the intuitive explanation provided by the original paper.

Thus, we can confidently train an optimal discriminator without worrying about the vanishing gradient. Referring again to *Equation 9*, the minimum discriminator loss of the two noisy distributions is

$$\begin{split} \min L_D(P_{r+\epsilon},P_{g+\epsilon}) &= \mathbb{E}_{x \sim P_{r + \epsilon}} [\log D^*(x)] - \mathbb{E}_{x \sim P_{g + \epsilon}} [\log (1-D^*(x))] \\ &= 2\log2- 2JS(P_{r+\epsilon}||P_{g+\epsilon})\end{split}$$
 
Of which $$P_{r+\epsilon}$$ and $$P_{g+\epsilon}$$ are the model distribution and the generative distribution after the introduction of noise. Thinking reversely, we can obtain the JS divergence between the two noisy distributions from the loss of the optimal discriminator, which is in a way the “distance” between them. “So now you mean that we can even indicate the progress of the training by discriminator loss? Is there really such a good thing?”

Unfortunately, no. Since the actual value of the JS divergence is affected by the variance of the noises, and as the annealing process starts to remove those noises, the newly obtained values are incomparable to the older ones. So, this is not an essential metric of the distance between the two distributions.

Due to the focus of this article, WGAN, we will not go further in depth about the noise-adding methodology proposed by the preliminaries on GAN. Again, interested readers may read the original paper for more details. The motivation of the noise addition only roots from the second problem with the original GAN, so although the training process is now stabilized, we still don’t have a reliable metric to indicate the training process. However, by rooting from the more fundamental first problem of GAN, WGAN replaces JS divergence with Wasserstein distance, which solves both problems at the same time.
The author, unfortunately, did not provide empirical results for this remedial solution.

**Part 3: The superior characteristics exhibited by the Wasserstein distance**

**Wasserstein distance**, aka Earth-Mover (EM) distance, is defined as below:
 
 $$W(P_r,P_g) = \inf_{\gamma \sim \Pi(P_r,P_g)} \mathbb{E}_{(x,y)\sim \gamma}[||x-y||]$$
 *(Equation 12)*
 
**Explanation:**   is the set of all possible combinations of the joint distributions of $$P_r$$ and $$P_g$$. Reversely, all marginal distribution in is either $$P_r$$ or $$P_g$$. And for all possible joint distributions, one could observe to obtain a model sample $$x$$ and a generative sample $$y$$, and thus calculate the distance. So, it is possible to take the expectation of this distance when sampling from the same joint distribution, hence. Taking the infimum of the expectations of all possible joint distributions, we obtain the Wasserstein distance.

Phew, that was a lot of jargons. Intuitively,   can be seen as the “cost” of moving this $$P_r$$ “sand pile” to the “position” $$P_g$$ by the path “$$y$$.” and   is exactly the “minimum cost” created by “the optimal path.” So that’s why it’s called “Earth-Mover” distance since it’s literal an earth mover.

The superiority of Wasserstein distance when compared to JS/KL divergence is that it still gives us an accurate measure of the distance between the two distributions even if they do not overlap. The original WGAN provided us with a simple example to illustrate this idea. Consider two univariate distributions $$P_1$$ and $$P_2$$, where $$P_1$$ is a uniform distribution on AB and $$P_2$$ is another uniform distribution on CD. A parameter $$\theta$$ is created to control the vertical distance between the two line segments.
 
 ![Imgur](https://i.imgur.com/J49BYSv.png)

Obviously,

$$\begin{split} KL(P_1||P_2) &= \begin{cases} +\infty  &\text{if} &\theta \neq 0  \\  0 & \text{if} & \theta=0 \end{cases}  \end{split}$$

*(jump discontinuity)*

$$\begin{split} JS(P_1||P_2) &= \begin{cases} \log2 &\text{if} &\theta \neq 0  \\  0 & \text{if} & \theta=0 \end{cases}  \end{split}$$
*(jump discontinuity)*

$$W(P_0,P_1)=|\theta|$$
*(smooth)*
   
While KL/JS divergences experience a huge jump, **Wasserstein distance is smooth and differentiable with respect to $$\theta$$**. If we would like to optimize this parameter by gradient descent, only the Wasserstein distance can provide useful information. Similarly, in a high-dimensional case, neither KL/JS divergence can provide information on distance, or could they work with gradient descent. 

_**But Wasserstein distance can!**_

**Part 4: From Wasserstein distance to WGAN**

*“So…. If we can define the generator loss as the Wasserstein distance, doesn’t that solve all the problems experience by the original GAN?”*

Well…. It’s not that simple, because according to the definition of W-1 distance *(Equation 12)*,  $$\inf_{\gamma \sim \Pi(P_r,P_g)}$$ is nonevaluative. But that’s fine, cause the author has provided us with an alternative form:
 
 $$W(P_r,P_g)=\frac{1}{K} \sup_{||f||_L \leq K} \mathbb{E}_{x \sim P_r}[f(x)]- \mathbb{E}_{x \sim P_g}[f(x)]$$
 *(Equation 13)*
 
How do we arrive at this? Well turns out that the process is too complicated that even the author himself threw the proofs to the appendix section in his original paper. So let’s just look at this from an intuitive perspective.

Before that there’s just one last concept that might be unfamiliar to the readers - Lipschitz continuity. Basically, it means that an additional constraint is added to a continuous function $$f$$ so that there exists a constant $$K$$ such that for any two elements $$x_1$$ and $$x_2$$ in the domain of the function, the following satisfies:
 
 $$|f(x_1)-f(x_2)| \leq K|x_1-x_2|$$
 
,$$K$$ is also referred to as the Lipschitz constant of $$f$$.

To put it more simply, if the domain of $$f$$ is all real numbers, the above is equivalent to restraining the absolute value of $$f$$’s derivative to be lower or equal to $$K$$. $$\log(x)$$ then is not Lipschitz continuous cause there isn’t an upper bound to its derivative. The Lipschitz continuous condition restrains the maximum rate of change of a continuous function.

Thus, Equation 13 means that as long as $$f$$’s Lipschitz constant no larger than $$K$$, we take the supremum of for all possible $$f$$, and then divided them by $$K$$. More specifically, we define a parametric class of the original function $$f$$ that is parameterized by the vector $$w$$, to which we assign the label $$f_w$$. Solving *Equation 13* yields:
 
 $$K \cdot W(P_r,P_g) \approx \max_{w:||f_w||_L \leq K} \mathbb{E}_{x \sim P_r} [f_w(x)] -\mathbb{E}_{x \sim P_g} [f_w(x)] $$
 *(Equation 14)*
 
**Now, the deep learning guy is going to be super excited to recognize something he/she’s been doing all along, cause we can replace the $$f_w$$ here by simply a generic neural network!** Since the approximative ability of neural networks is enormous, it is safe to assume that although the entire class of $$f_w$$’s cannot cover all possibilities of $$f$$, it still provides a good approximation to the $$\sup_{||f||_L \leq K}$$ term in *Equation 13$.

Finally, we cannot forget that $$||f||_L$$  in *Equation 14* still needs to be smaller than $$K$$. Again, we don’t care what value it takes on, as long as it’s not positive infinity. The reason is that enlarging $$K$$ will only enlarge the gradient, while the direction of which will remain the same. So, the author added a simple procedure to ensure the value of $$K$$: cropping all parameters so that they reside in some range like.  Consequently, the partial derivative $$\frac{\partial f_w}{\partial x}$$ of an arbitrary input $$x$$ will also be restrained to some unknown range that is governed by an unknown constant $$K$$, achieving the Lipschitz continuous condition. In the actual implementation, we need to clip the parameters $$w$$ back to the range after each update.

**At this point, we can create a discriminative neural network that is parameterized by $w$ and has an affine function for the activation function in the last layer. By restricting elements in w to a specific range, we optimize**
 
 $$L = \mathbb{E}_{x \sim P_r} [f_w(x)] -\mathbb{E}_{x \sim P_g} [f_w(x)]$$
  *(Equation 15)*
  
**To its maximum,  $$L$$ will be a good approximation of the Wasserstein distance between the model distribution and the generative distribution (ignoring the constant $$K$$). Since the original GAN discriminator is tackling a binary classification problem, the last layer has to use a sigmoid function to normalize the results. But now, we have WGAN using discriminator fw approximating the Wasserstein distance, which is apparently a regression problem, still keeping the sigmoid layer doesn’t make sense.**

**And to approximately minimize the Wasserstein distance, we can minimize L instead. The vanishing gradient is no longer a problem due to the superior W-1 distance. And since the first term of $L$ is not related to the generator, we can obtain similarly two loss functions used in WGAN:**
  
$$-\mathbb{E}_{x \sim P_g} [f_w(x)]$$
  *(Equation 16, WGAN Generative network loss function)*
  $$-\mathbb{E}_{x \sim P_r} [f_w(x)] +\mathbb{E}_{x \sim P_g} [f_w(x)]$$
  *(Equation 17, WGAN Discriminative network loss function)*
  
Equation 15 is the additive reciprocal of Equation 17, which can be used to indicate the training process. A smaller value indicates a smaller Wasserstein distance between the model distribution and the generative distribution, hence a better GAN network.

Here's just a re-post of the pseudo-code:
![Imgur](https://i.imgur.com/y7U95UI.jpg)
 

As mentioned earlier in this article, there were only four modifications that WGAN made on the basic GAN framework:
1.    Removing the sigmoid activation function in the last layer
2.    Removing the log in the loss function of G/D models
3.    Clipping the parameters so that their absolute values are smaller than a constant c
4.    Not using any optimization methods based on momentum (like the naïve method and Adam); RMSProp is the recommended method, though SGD works as well

The first three are obtained from theoretical analyses and are already introduced above. The fourth one is more of a trick since only empirical results can support the claim. The author noticed that if Adam is used for gradient optimization, the loss of discriminator will sometimes go crazy, and whenever it does, the update Adam provides are the opposite of the gradient direction, which suggests an unstable loss gradient. So, the author concludes that momentum-based optimization methods are not suitable for WGAN and encourages the use of RMSProp method, due to its adaptive ability against unstable gradients.
The author did provide huge amounts of empirical results for his WGAN, and this article will present the most important three. 

Firstly, Wasserstein estimate is highly correlated with the quality of the picture:
 
 ![Imgur](https://i.imgur.com/rwseEZe.png)

Secondly, when using a DCGAN architecture, WGAN achieves similar results:

![Imgur](https://i.imgur.com/33Xw1g5.png)

But what is really amazing about WGAN is that it still functions pretty well even without the DCGAN architecture. For example, when batch normalization is no longer used, DCGAN fails *miserably*:
![Imgur](https://i.imgur.com/6mqZW1F.png)

And when WGAN and GAN use multilayer perceptron (MLP) for their G/D networks (no CNN), WGAN do experience a slight decrease in its quality. But for the original GAN, not only the decrease is more drastic, but it also experiences from mode collapse, where the lack of diversity is evident.
![Imgur](https://i.imgur.com/J8kxMU9.png)

Thirdly, no collapse mode was observed during all WGAN training sessions, to which the author claims that the problem is somewhat solved.

The last point of nuance, something that the original paper didn’t mention, is that the author claimed that the approximated Wasserstein distance could guide researchers to adjust hyperparameters of the network when compared to different sessions. I believe that one needs to be careful when doing such comparisons since the error of Wasserstein distance estimation differs from session to session. **The layers/nodes of the discriminator, the training epochs, there are really lots of doubts on how comparable the two sessions really are regarding the Wasserstein distance updates.**

*(Someone in the comment section points out that **alteration on the hyperparameters of the discriminator will directly affect the Lipschitz constant $K$, which renders future sessions incomparable to the present and past ones.** This is something that one should really pay attention to when implementing WGAN. To this, I came up with a more engineering-wise and less elegant solution. Take the same generative/discriminative distribution with different discriminators; train them separately until convergence, and then look at the difference between the metric. This difference can be viewed as the ratio between the Lipschitz constants of the two sessions, which can be then used to correct the metrics in later sessions)*

**Section 5: Conclusion**

Preliminaries on WGAN analyzed the problem with the two forms of original GAN proposed by Ian Goodfellow. The first form has an equivalent form of minimizing the JS divergence when an optimal discriminator is obtained, and since a randomly initialized distribution is highly likely to overlap significantly with the model distribution, the jump discontinuity of JS divergence causes the gradient to vanish. The second form has an absurd goal of minimizing KL divergence while maximizing JS divergence when the optimal discriminator is obtained, which destabilizes the gradient, and the asymmetric nature of KL divergence caused the collapse mode in generative networks.

Preliminaries on WGAN nevertheless offered a remedial solution to counter the overlap problem in the original GAN. By adding noises to the distributions, the forced overlaps can stabilize the training process, where obtaining an optimal discriminator is no longer problematic. However, no empirical results are presented.
The original GAN paper introduced the concept of Wasserstein distance, which has a desiring ability to maintain its smoothness even when the two distributions do not overlap. Opposed to the problematic KL/JD divergence, Wasserstein distance theoretically solves the problem of vanishing gradients. Then, mathematical transformations made Wasserstein distance evaluable, and a neural network is capable of approximating this metric. Whenever an approximately optimal discriminator is obtained, minimizing the Wasserstein distance can pull the two distributions together. Not only does WGAN make the training process stable, it provides a useful indicator for its training process, which is also highly correlated with the quality of the generated examples. The author offered extensive empirical results for the WGAN framework.

