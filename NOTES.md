# COS 511: Theoretical Machine Learning

## Lecture #1
The first part of Lecture 1 just explains the basics of what machine learning is (classification, regression, behavior-learning). The notes go into the goals of machine learning research: efficiency (time, space, amount of data required by learning algorithms), generality of learning, accuracy, and interpretability of prediction rules.

The goal of theoretical machine learning is to derive insights/intuitions that may be helpful with designing practical algorithms or understand learning problems. The course mentions that it will cover topics such as the number of examples necessary to learn, theoretical understanding of boosting and SVMs, online learning, estimating probability distributions, and game theory.

The notes then go into an example of a classification learning problem to demonstrate that more data eliminates more potential models for the pattern, and that the simplest models (and obviously, most accurate) are most probable. The latter rule is referred to as "Occam's razor".

Finally, Schapire begins to formalize the idea of a learning model (simplified to study its mathematical behavior). The goals of such a model include what is being learned, the origin of data, how it's being learned, and the goal of learning.

- Example (or instance): Object being classified
- Attributes (or features, variables, dimensions): Description of example
- Label (or class): Category being predicted (often simplified to 0,1)
- Domain Space (or instance space): All possible examples
- Concept: Mapping from examples to labels (of the form c : X -> {0,1}) 
- Concept Class: A collection of concepts

Often, we will assume examples have been labelled by an unknown concept from a known concept class (i.e ground truth that we are trying to find/approximate). 

The first learning model covered by the notes is the consistency model (unrealistic but intuitive). In the consistency model, a concept class is learnable if there exists an algorithm that, when given any set of labeled examples, finds a concept consistent with the examples (or correctly states there is no such concept).

## Lecture #2
### Consistency Model
Lecture 2 covers concept classes that are and aren't learnable under the consistency model.

First, Schapire brings up boolean logic with the instance space of n-bit vectors. Suppose our concept class is the set of all monotone conjunctions (AND of boolean variables), mappings from boolean variables to labels. An algorithm to always find a consistent monotone conjunction is this: only consider positive examples and keep columns where entries are all 1, and then tries this mapping on negative examples, and if they're consistent return the conjunction.

With this algorithm, only subsets of the returned conjunction are conjunctions that are consistent with all positive examples (boolean variables not in the returned conjunction are 0 for one positive label at least). This is the set of all concepts consistent with positive examples.

Now we want a concept consistent with all negative examples. Note that if any negative example is consistent with a single boolean variable, it is consistent with the monotone conjunction of all selected boolean variables. Thus if the monotone conjunction of all selected variables is not consistent with the negative examples, then none of the concepts consistent with positive examples is consistent with negative examples, so no consistent concept exists. 

Therefore the concept class of monotone conjunctions is learnable.

For the concept class of monotone disjunctions (the OR of boolean variables), we can convert the examples into conjunction form using De Morgan's rule and reduce to learning the monotone disjunctions concept class.

Now we consider the concept class of the set of conjunctions, whereby the boolean variables can be negated. We can reduce this to the monotone conjunction problem by adding a new set of variables equivalent to the inverse of all boolean variables.

Schapire then gets into examples from geometry. We first look at the axis-aligned rectangles problem: given a set of labelled points, find an axis-aligned rectangle (concept) such that all the points inside are +, and - otherwise. An algorithm is to scan through all positive examples and use the largest corners to define the smallest enclosing rectangle. If any negative examples are in this rectangle then it's not learnable, otherwise output the rectangle. 

Now we consider the concept class of half-spaces or linear threshold functions. Each concept here is a boolean function where points on one side of a linear hyperplane are labeled +, and - on the other side. We can formulate finding the hyperplane as a linear programming solution, so the concept class of half-spaces is learnable.

The notes then get into more boolean logic examples (kCNF, 2CNF, general CNF, which we skip).

Problems with the consistency model is its lack of explaining how the concept generalizes to new data. Thus it really says little about learning at all, so we need to find a new model.

### PAC Learning Model
The goal of the PAC learning model is to develop a hypothesis with as little error as possible, rather than finding ground truth. We will answer where the examples will come from in this model. 

We define error as the probability that any given example x (which comes from an unknown target distribution that generates x), evaluating the hypothesis h(x) is not equivalent to the label of the consistent concept c(x).

The goal of the learning algorithm is to minimize the error for the testing dataset.

## Lecture #3

