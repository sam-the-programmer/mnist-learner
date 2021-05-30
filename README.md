# MNIST Learner

This MNIST learner is a deep artificial neural network (ANN) that uses the `tensoflow.keras` API
in *Python*. The final aim of the project will be to compare ANNs using dense layers, recurrent
networks and convolutional networks to aim to read >97% accuracy on validation and testing.

Presently, this is under heavy developement, but so far, this is the state of the project:

 - Functioning training loop with accuracy of 91% in validation (95% in training), but this varies
 - Efficient data parsing and manipulation
 - Only have a dense layer model
 - Not yet parsed testing data
 - Not yet generated confusion matrices

------
<br/>

## Network Architectures

### Dense

**Input:** 784 inputs

**Layer 1:** 16 neurons

**Layer 2:** 16 neurons

**Output:** 10 outputs

**Loss:** Sparse Categorical Cross Entropy
