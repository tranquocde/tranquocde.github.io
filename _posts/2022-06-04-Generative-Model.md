---
layout: post
title: "Generative Models"
background: ''
---
# Generative Model

[slide](http://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L4.pdf)

[code](https://colab.research.google.com/drive/1MNLBGtLv_segHZpY68ClkOQ39myrfgJC)

## Generative Modeling

- **Goal**:
    - Take as input training samples from some distribution and learn a model that represents that distribution.
        
        ![Screenshot 2023-05-26 at 10.39.55.png](/img/posts/Generative/Screenshot_2023-05-26_at_10.39.55.png)
        

→ How can we learn $P_{model}(x)$ similar to $P_{data}(x)$?

- Why  generative models ?
    - Capable of uncovering **underlying features** in a dataset

**Latent variable**

- A variable that can only be inferred indirectly through a mathematical model from other **observable variables** that can be directly [observed](https://en.wikipedia.org/wiki/Observation) or [measured](https://en.wikipedia.org/wiki/Measurement)

## Auto-encoders

- Background:
    - Unsupervised approach for learning a lower-dimensional feature representation
    - Train the model to use these features to **reconstruct the original data**
        
        ![img](img/posts/Generative/Screenshot_2023-05-26_at_10.48.21.png)
        

## Variational Auto-encoders (VAE)

A key difference with traditional auto-encoder:

- Variational auto-encoders are a probabilistic twist on auto-encoders
- Sample from the mean and standard deviation to compute latent sample
    
    ![Screenshot 2023-05-26 at 10.51.55.png](Generative%20Model%20611f0566cb8b4703affa8e6c80a5515d/Screenshot_2023-05-26_at_10.51.55.png)
    
- $L(\phi,\theta,x)$ = (reconstruction loss) + (regularization term)
    
    ![Screenshot 2023-05-26 at 10.59.24.png](Generative%20Model%20611f0566cb8b4703affa8e6c80a5515d/Screenshot_2023-05-26_at_10.59.24.png)
    
- Intuition on regularization:
    - What properties do we want to achieve from regularization?
        - Continuity: Points that are closer in latent space ⇒ similar content after decoding
        - Completeness: Sampling from latent space ⇒ meaningful content after decoding

**Problem**: We can not backpropagate gradients through sampling layers! 

**⇒ Reparametrizing the sampling layer !** 

![Screenshot 2023-05-26 at 10.58.01.png](Generative%20Model%20611f0566cb8b4703affa8e6c80a5515d/Screenshot_2023-05-26_at_10.58.01.png)

![Screenshot 2023-05-26 at 10.57.34.png](Generative%20Model%20611f0566cb8b4703affa8e6c80a5515d/Screenshot_2023-05-26_at_10.57.34.png)

## Generative Adversarial Networks (GAN)