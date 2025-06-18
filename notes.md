# Gaussian Process for CLIP adaptation

## Problem definition

Let us consider a classification problem with **K** classes, each represented by a prototype vector **t<sub>k</sub> ∈ ℝ<sup>d</sup>**, which is expressed as a linear combination of fixed text embeddings  

t<sub>k</sub> = Σ<sub>j=1…M</sub> α<sub>k,j</sub> f<sub>j</sub> = **F α<sub>k</sub>**

where **F = [f₁,…,f_M] ∈ ℝ<sup>d×M</sup>** is the matrix of fixed text embeddings obtained from the CLIP text encoder and **α<sub>k</sub> ∈ ℝ<sup>M</sup>** are the unknown balancing weights for class *k*.

**Adaptation.** Given a support set  

S = {(x<sup>(m)</sup>, y<sup>(m)</sup>)}<sub>m=1…K × N</sub>

of *N* images per class and one-hot labels y<sub>i</sub> ∈ {0,1}<sup>K</sup>, a pretrained CLIP visual encoder produces visual embeddings **v<sub>i</sub> ∈ ℝ<sup>d</sup>**.  
We learn a linear projection **W ∈ ℝ<sup>d×d</sup>** and compute  

ṽ<sub>i</sub> = **W** v<sub>i</sub>.

Logits for class *k*:

ℓ<sub>i,k</sub> = ṽ<sub>i</sub><sup>⊤</sup> t<sub>k</sub> = ṽ<sub>i</sub><sup>⊤</sup>(**F α<sub>k</sub>**)

Class probabilities:

ŷ<sub>i,k</sub> = exp (ℓ<sub>i,k</sub>) / Σ<sub>j=1…K</sub> exp (ℓ<sub>i,j</sub>)

Cross-entropy over the support set S:

L<sub>CE</sub>(x<sub>i</sub>, y<sub>i</sub>) = −Σ<sub>k=1…K</sub> y<sub>i,k</sub> log ŷ<sub>i,k</sub>

## Gaussian Process Prior on Prototypes

Model α<sub>k</sub>(f) as a latent function over text embeddings f ∈ ℝ<sup>d</sup> with a Gaussian Process prior:

α<sub>k</sub>(f) ~ 𝒢𝒫(m<sub>k</sub>(f), K<sub>k</sub>(f,f′))

Evaluated at {f<sub>j</sub>}<sub>j=1…M</sub> we obtain

p(α<sub>k</sub>) = 𝒩(m<sub>k</sub>, K<sub>k</sub>)  

m<sub>k</sub> = [m<sub>k</sub>(f₁),…,m<sub>k</sub>(f_M)]<sup>⊤</sup>, (K<sub>k</sub>)<sub>ij</sub> = K<sub>k</sub>(f<sub>i</sub>, f<sub>j</sub>)

## Posterior Inference

Joint likelihood:

p(y | {α<sub>k</sub>}, W,{v<sub>i</sub>}) = Π<sub>i=1…N</sub> exp(y<sub>i</sub><sup>⊤</sup>ℓ<sub>i</sub>) / Σ<sub>k=1…K</sub> exp (ℓ<sub>i,k</sub>)

Bayes posterior (intractable):

p({α<sub>k</sub>} | {(x<sub>i</sub>, y<sub>i</sub>)}, W) ∝ p(y | {α<sub>k</sub>}, W,{v<sub>i</sub>}) ⋅ Π<sub>k=1…K</sub> p(α<sub>k</sub>)

### Variational approximation

Assume q(α<sub>k</sub>) = 𝒩(μ<sub>k</sub>, Σ<sub>k</sub>) and optimize the ELBO

ℒ = E<sub>q</sub>[log p(y | {α<sub>k</sub>}, W)] − Σ<sub>k=1…K</sub> KL(q(α<sub>k</sub>) ‖ p(α<sub>k</sub>))

Monte-Carlo approximation with M samples α<sub>k</sub><sup>(m)</sup>:

−ℒ ≈ −(1/M) Σ<sub>m=1…M</sub> log p(y | {α<sub>k</sub><sup>(m)</sup>}, W) + β Σ<sub>k=1…K</sub> KL(q(α<sub>k</sub>)‖p(α<sub>k</sub>))
