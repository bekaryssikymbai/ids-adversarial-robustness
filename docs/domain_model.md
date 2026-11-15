# Domain Model: Machine Learning Formalization

## 1. Problem Definition
Let \(X \in \mathbb{R}^{n \times d}\) be the feature matrix of network traffic, where \(n\) is the number of samples and \(d\) is the number of features. Let \(y \in \{0, 1\}^n\) be the binary target vector.

## 2. Loss Function
The objective is to minimize the regularized loss function:
\[
 \mathcal{L}(\theta) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k) 
\]
where \(l\) is the Cross-Entropy loss.

## 3. Adversarial Attack Formulation (FGSM)
The FGSM generates adversarial examples \(X^{adv}\) by adding a perturbation:
\[
 X^{adv} = X + \epsilon \cdot \text{sign}(\nabla_X \mathcal{L}(\theta, X, y)) 
\]
where \(\epsilon\) is the attack strength.

## 4. Evaluation Metric (F1-Score)
\[
 F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} 
\]
