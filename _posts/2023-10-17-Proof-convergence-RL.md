---
layout: post
title: "Proof for optimal convergent of Value Iteration using contraction"
author: "tranquocde"
background: "/img/posts/Proof-RL/contraction.png"
---

<!-- # Proof for optimal convergent of Value Iteration using contraction -->

# **Note**:

 In this post, I assume the reader already got some very basic ideas in Reinforcement Learning , and had interest in Math :D  ( Personally, I think RL is the subject with tons of Math ,and is the hardest one but very interesting and promising ). I will explain the very most basic and important aspect of RL in other posts. I just wrote this post after understand everything about this very interesting proof. 

# **Notation**:

- **Model** : The mathematical description of the dynamics and rewards of the agent’s environment ( we can think the environment like a maze with traps and obstacles ). It also includes the transition probabilities $P(s'\|s,a)$ , which is the prob. to reach the state s’ from state s if you take the specific action a ( we simulate the world with lots of uncertainty :D). The state s belongs the S which is the universal state set (include all possible states) , the action a in some case can depend on s but generally it belongs to A - the set of all actions ( this A can depend on state s in some specific problem)
- **Reward** : There are a lot of way to define a reward, but in this post, we just define it as immediate reward , $r(s,a)$, the reward the agent get after taking action $a$ at state $s$.
- **Policy** : A mapping $\pi : S \rightarrow A$ maps the agent’s state to its action ( kinda a strategy for the agent to follow ). Policy can be stochastic or deterministic.
- **Value function** : The value function $V^{\pi}(s)$  associated with specific policy $\pi$ at state s, is the cumulative sum of  future (discounted) rewards obtained from state s following the policy $\pi$. ( Why discounted?  It is only good for infinite state (or so many for handle), I would like to explain it deeper in other post )

## **Recall** : < Value Iteration >

- Set k = 1
- Initialize $V_o(s)$= 0 for all state s
- Loop until convergence :
    - For each state s: <update V-value function>
        
        $$
        V_{k+1}(s) = \max_a R(s,a) + \gamma\sum_{s' \in S}P(s'|s,a)V_k(s')
        $$
        
        update policy:
        
        $$
        \pi_{k+1}(s) = \arg \max_aR(s,a) + \gamma \sum_{s' \in S}P(s|s,a)V_k(s')
        $$
        

# Bellman backup operator :

For element $ U \in R^{\|S\|} $ , **Bellman backup operator $B^{\pi}$  for a particular policy $\pi$** at state s is defined as : 

$$
B^{\pi}U(s) = R^{\pi}(s) + \gamma\sum_{s'\in S}P^{\pi}(s'|s)U(s), \forall s\in S 
$$

### Bellman backup operator for optimal policy

Given the MDP <Markov Decision Process> = (S,A,P,R,$\gamma$)

Given U is an element in $R^{\|S\|}$, **the Bellman optimality backup operator $B^*$** is defined as : 

$$
(B^*U)(s)= \max_{a\in A}R(s,a) + \gamma \sum_{s' \in S}P(s'|s.a)U(s'), \forall s\in S
$$

⇒ We can rewrite the value iteration algorithm in term of Bellman operator:

$$
V_{k+1} = B^*V_k ;
$$

$$
\pi_{k+1}(s) = \arg \max_aR(s,a) + \gamma \sum_{s' \in S}P(s|s,a)V_k(s')
$$

⇒ How to prove that optimal policy $ \pi^* $ is the fixed point of $B^*$ ???

## Contraction

**Informal definition** : A contraction is **a transformation T that reduces the distance between every pair of points**. That is, there is a number r < 1 with : dist(T(x, y), T(x', y')) ≤  r * dist((x, y), (x', y')) for all pairs of points (x, y) and (x', y').

> We call $B$ is a contraction if B is the mapping $R^d \rightarrow R^d$ and there is a number $\gamma$ (0≤$\gamma$<1) satisfies:  for and any distinct pair of U,V in $R^d$ we have $BU - BV ≤ \gamma (U-V)$
> 

# Note:

Fixed point of a contraction is unique

## Proof idea: 

- Prove $B^*$ is a contraction
- Using the property of Cauchy sequence for a contraction
- Using the property of uniqueness of fixed point of a contraction
- Show that if $\mu$ is an optimal policy then $ (B^* V^\mu)(s) = V^{\mu}(s),\forall s \in S $, from that implies $V^{\mu}$ is the fixed point of $B^*$
    - using the definition of$B^*$ and the Bellman’s formula for $V^{\mu}(s)$.
- From that, due to the convergence of Cauchy series then the Value-iteration Algorithm converges at the fixed point of $B^*$ ,  which is $V^{\mu}$ and policy $\pi_i$ will converge to the optimal policy $\mu$