# CM3015 Machine Learning and Neural Networks - Deep Learning Fundamentals (DLWP 1-4 Condensed)

This document condenses key concepts from "Deep Learning with Python" Chapters 1-4, relevant to the CM3015 coursework focusing on the MNIST dataset and adherence to the DLWP 4.5 workflow.

## Chapter 1: What is Deep Learning?

### Artificial Intelligence, Machine Learning, and Deep Learning

*   **Artificial Intelligence (AI):** Automate intellectual tasks normally performed by humans.
*   **Machine Learning (ML):**  Algorithms that learn rules from data instead of explicit programming.  A system is *trained* rather than programmed.
*   **Deep Learning (DL):**  A subfield of ML emphasizing learning successive layers of increasingly meaningful representations.  The "deep" refers to the depth (number of layers) of the model.  Often implemented with neural networks.
*   **Classical Programming vs. Machine Learning:**  Classical programming inputs rules and data to output answers; machine learning inputs data and answers to output rules.
*   **Representation Learning:** The central problem in ML/DL is to meaningfully transform data into useful representations.  A *representation* is a different way to look at or encode data.

### The "Deep" in Deep Learning

*   Emphasizes successive layers of representations.
*   *Depth* of the model: How many layers contribute.
*   Representations are often learned via *neural networks*.
*   Deep learning: multistage way to learn data representations.

### How Deep Learning Works

1.  **Weights:** Layers are parameterized by *weights* (numbers).
2.  **Loss Function:** Measures how far the network's output is from the expected target (the *objective function*).
3.  **Optimizer (Backpropagation):** Uses the loss score as a feedback signal to adjust the weights, decreasing the loss.
4.  **Training Loop:** Repeat (1-3) until weights minimize the loss function.

### What Deep Learning Has Achieved

*   Near-human-level performance in image classification, speech recognition, handwriting transcription, machine translation, text-to-speech, digital assistants, autonomous driving, ad targeting, web search, natural language question answering, Go.

### Promise of AI

*   AI will be your assistant, even your friend; it will answer your questions, help educate your kids, and watch over your health.

### Before Deep Learning: A Brief History of Machine Learning

*   **Probabilistic Modeling:** Naive Bayes, Logistic Regression.
*   **Early Neural Networks:** LeNet (Yann LeCun, 1989) for handwritten digit classification.
*   **Kernel Methods:** Support Vector Machines (SVMs). Finds optimal decision boundaries.
    *   *Kernel Trick*: Computes distances between points in a new representation space using a kernel function.
*   **Decision Trees, Random Forests, Gradient Boosting Machines:** Ensembling weak prediction models (decision trees). XGBoost.
*   **Return to Neural Networks:**  Breakthroughs around 2010 (Hinton, Bengio, LeCun, IDSIA). ImageNet challenge (2012).

### What Makes Deep Learning Different

*   Automated *feature engineering* (no manual crafting of representations).
*   Learns all layers of representation *jointly*, at the same time.

### Why Deep Learning Now?

*   **Hardware:**  Faster CPUs, GPUs.
*   **Data:**  Internet made large datasets available.  ImageNet.
*   **Algorithms:**  Better gradient propagation techniques (activation functions, weight initialization, optimizers).
*   **Investment:** Industry investment far beyond anything previously seen in the history of AI.

## Chapter 2: Before We Begin: The Mathematical Building Blocks of Neural Networks

### A First Look at a Neural Network

*   Classification of handwritten digits (MNIST) using Keras.
*   Data (image pixels) reshaped and scaled.
*   `Sequential` model with `Dense` layers.
*   `softmax` layer for multiclass output (probability scores).
*   Compilation: `optimizer` (rmsprop), `loss` (categorical_crossentropy), `metrics` (accuracy).
*   `fit` method: trains the model.
*   `evaluate` method: tests the model.
*   *Overfitting*: test accuracy lower than training accuracy.

### Data Representations for Neural Networks

*   **Tensors:** Fundamental data structure for all machine-learning systems.
    *   0D (Scalar): Single number.
    *   1D (Vector): Array of numbers.
    *   2D (Matrix): Array of vectors.
    *   3D/Higher: Array of matrices, etc.
*   **Key Tensor Attributes:**
    *   *Number of axes (rank/ndim)*
    *   *Shape* (tuple of integers)
    *   *Data type (dtype)* (e.g., float32, uint8)
*   **Tensor Slicing:** Selecting specific elements.
*   **Data Batches:** Models process data in small batches (first axis is the *samples axis/batch dimension*).

### Real-World Examples of Data Tensors

*   **Vector Data:** (samples, features) - tabular data.
*   **Timeseries/Sequence Data:** (samples, timesteps, features) - stock prices, text.
*   **Images:** (samples, height, width, channels) or (samples, channels, height, width). TensorFlow uses channels-last; Theano uses channels-first.
*   **Video:** (samples, frames, height, width, channels).

### The Gears of Neural Networks: Tensor Operations

*   All learned transformations can be reduced to tensor operations.
*   **Element-wise Operations:**  Applied independently to each entry (e.g., ReLU, addition).  Highly parallelizable.
*   **Broadcasting:**  Smaller tensors automatically expanded to match the shape of larger tensors during element-wise operations.
*   **Tensor Dot (Tensor Product):** Combines entries (not element-wise).  `np.dot(x, y)`. Requires compatible shapes: `x.shape[1] == y.shape[0]`.
*   **Tensor Reshaping:**  Rearranging rows/columns to match a target shape. `x.reshape((new_shape))`.  Includes *transposition*.

### The Engine of Neural Networks: Gradient-Based Optimization

*   Layers transform data: `output = relu(dot(W, input) + b)`
    *   `W`, `b`: Layer's weights/trainable parameters.  Learned from data.  Initialized randomly.
*   **Training Loop:**
    1.  Draw batch of training samples `x` and targets `y`.
    2.  Run network on `x` (forward pass) to get predictions `y_pred`.
    3.  Compute loss between `y_pred` and `y`.
    4.  Update weights to reduce the loss.
*   Goal:  Low loss on training data.  Network "learns" to map inputs to targets.

### Derivatives and Gradients

*   *Derivative*: The slope of the local linear approximation of a function.  Indicates how `f(x)` changes as `x` changes.
*   *Gradient*: Derivative of a *tensor operation*.
*   Move weights in the *opposite* direction of the gradient to *decrease* the loss.  `W1 = W0 - step * gradient(f)(W0)`. `step` is a small scaling factor.

### Stochastic Gradient Descent (SGD)

*   Analytical solution (gradient(f)(W) = 0) is intractable.
*   SGD Algorithm:
    1.  Draw a batch of training samples `x` and corresponding targets `y`.
    2.  Run the network on `x` to obtain predictions `y_pred`.
    3.  Compute the loss of the network on the batch, a measure of the mismatch between `y_pred` and `y`.
    4.  Compute the gradient of the loss with regard to the network's parameters (a backward pass).
    5.  Move the parameters a little in the opposite direction from the gradient—for example `W -= step * gradient`—thus reducing the loss on the batch a bit.
*   *Mini-batch SGD*: Each batch of data is drawn at random.
*   *Learning rate*: Step factor.
*   Optimization Methods/Optimizers: Variants of SGD (e.g., SGD with momentum, Adagrad, RMSProp, Adam).
*   *Momentum*: Addresses convergence speed and local minima.  Simulates a ball rolling down the loss curve.

### Backpropagation Algorithm

*   Calculates the gradient values.
*   *Chain Rule*: `f(g(x)) = f'(g(x)) * g'(x)`. Used to derive complex derivatives from simple operations.
*   Modern frameworks (TensorFlow) use *symbolic differentiation* to compute gradient functions automatically.

## Chapter 3: Getting Started with Neural Networks

### Anatomy of a Neural Network

*   **Layers:** Data-processing modules.  Have weights (learned tensors). Can be stateless or stateful.
*   **Network (Model):** Directed acyclic graph of layers. Common types: linear stack, two-branch, multihead, Inception blocks. Network topology defines the hypothesis space.
*   **Loss Function (Objective Function):** Quantity to be minimized during training. Represents a measure of success.
*   **Optimizer:** Determines how the network will be updated based on the loss function (implements SGD).

### Keras Introduction

*   High-level, user-friendly deep learning framework for Python.
*   Supports CPU and GPU.
*   Three backends: TensorFlow, Theano, CNTK.
*   Modular design.

### Keras Workflow

1.  Define training data (inputs and targets).
2.  Define a network of layers (model) mapping inputs to targets.
3.  Configure the learning process: loss function, optimizer, metrics.
4.  Iterate on training data: `model.fit()`.

### Keras Model Definition

*   `Sequential` class (linear stacks of layers).
*   Functional API (for directed acyclic graphs).
*   Example (Sequential):

    ```python
    from keras import models
    from keras import layers
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
    model.add(layers.Dense(10, activation='softmax'))
    ```

### Model Compilation

```python
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='mse',
              metrics=['accuracy'])
```

### Model Training

```python
model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)
```

### Classifying Movie Reviews: A Binary Classification Example

*   IMDB Dataset: 50,000 polarized reviews (25k train, 25k test).
*   Reviews: Sequences of integers (word indices).
*   Labels: 0 (negative), 1 (positive).
*   **Data Preparation:** Vectorize lists into tensors. One-hot encoding.
*   **Model Building:** Stack of `Dense` layers with `relu` activations.
    *   Hidden units: Dimensions in representation space. Balance capacity.
    *   Output layer: `Dense(1, activation='sigmoid')` for probability output.
    *   Loss function: `binary_crossentropy`.
    *   Optimizer: `rmsprop`.
*   **Validation:** Monitor accuracy on validation set to prevent overfitting.
*   **Prediction:** `model.predict(x_test)` to generate likelihood of reviews being positive.

### Classifying Newswires: A Multiclass Classification Example

*   Reuters Dataset: Newswires and their topics (46 classes).
*   Single-label, multiclass classification.
*   One-Hot Encoding: Important for categorical data, label -> all-zero vector, 1 in place of label index.
*   softmax activation for probability distribution over output classes.
*   `categorical_crossentropy` loss function.
*   Avoid information bottlenecks - ensure the dimensions are large enough
*   `sparse_categorical_crossentropy` can be used for integer encoded labels.

### Predicting House Prices: A Regression Example

*   Boston Housing Price Dataset: Median price of homes in Boston suburbs.  Few data points (506).  Features have different scales.
*   **Data Preparation:** Feature-wise normalization (subtract mean, divide by standard deviation).
*   **Model Building:** Small network with two hidden layers.  Linear last layer (no activation).
    *   Loss function: `mse` (Mean Squared Error).
    *   Metric: `mae` (Mean Absolute Error).
*   **K-Fold Cross-Validation:** Use when data is limited.  Average K scores.
*   **Results:** Evaluate trained model.

## Chapter 4: Fundamentals of Machine Learning

### Four Branches of Machine Learning

*   **Supervised Learning:** Learn to map inputs to known targets (classification, regression).
*   **Unsupervised Learning:** Find transformations without targets (dimensionality reduction, clustering).
*   **Self-Supervised Learning:** Supervised without human-annotated labels (autoencoders, predicting next frame/word).
*   **Reinforcement Learning:** Agent learns actions to maximize reward.

### Key Supervised Learning Terminology

*   **Sample/Input:** One data point.
*   **Prediction/Output:** What the model outputs.
*   **Target:** Ground truth.
*   **Loss Value:** Distance between prediction and target.
*   **Classes:** Possible categories.
*   **Label:** Specific class annotation.
*   **Ground-truth/Annotations:** Targets for a dataset.
*   **Binary Classification:** Two exclusive categories.
*   **Multiclass Classification:** More than two categories.
*   **Multilabel Classification:** Sample assigned multiple labels.
*   **Scalar Regression:** Target is a continuous scalar value.
*   **Vector Regression:** Target is a continuous vector.
*   **Mini-batch:** Small set of samples processed simultaneously.

### Evaluating Machine Learning Models

*   Training set, validation set, test set (prevent information leaks/overfitting to validation data).
*   **Evaluation protocols:**
    *   Hold-out validation
    *   K-fold cross-validation
    *   Iterated K-fold validation with shuffling
*   **Things to keep in mind:** Data representativeness, the arrow of time (time-series data), redundancy in data.

### Data Preprocessing, Feature Engineering, and Feature Learning

*   **Data Preprocessing:**
    *   *Vectorization*: Turn data into tensors.
    *   *Value normalization*: Scale values to a small range (e.g., 0-1, -1 to 1) with feature-wise centering.

*   **Feature Engineering:** Using domain knowledge to improve algorithm performance by transforming data before input. Less necessary with DL.

### Overfitting and Underfitting

*   *Overfitting*: Model learns patterns specific to training data that don't generalize.
*   *Underfitting*: Model hasn't learned all relevant patterns.
*   **Regularization:** Techniques to prevent overfitting.

### Common Regularization Techniques

*   **Reducing the Network's Size:** Fewer layers, fewer units per layer.
*   **Adding Weight Regularization:** L1 (absolute value), L2 (weight decay, square value) of weights.
*   **Adding Dropout:** Randomly dropping out (zeroing) output features during training.

### The Universal Workflow of Machine Learning

1.  **Define the problem and assemble a dataset:** What will your input data be? What are you trying to predict? Make data available and annotate with labels. Consider if output can be predicted from inputs. Hypothesize available data is sufficiently informative.
2.  **Choose a measure of success:** Accuracy? Precision and recall? Customer-retention rate? Metric for success should directly align with your higher-level goals, such as the success of your business.
3.  **Decide on an evaluation protocol:** Maintaining a hold-out validation set, doing K-fold cross-validation, or doing iterated K-fold validation.
4.  **Prepare your data:** Format as tensors, values should be scaled to small values and normalize features
5.  **Develop a model that does better than a baseline:** Establish statistical power, make key choices: last-layer activation, loss function, optimization configuration.
6.  **Scale up: develop a model that overfits:**  Add layers, make the layers bigger, train for more epochs. When validation performance degrades, you’ve achieved overfitting.
7.  **Regularize your model and tune your hyperparameters:**  Add dropout, different architectures, L1 and/or L2 regularization.
