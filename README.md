# MNIST Handwritten Digit Classifier (DLWP)

This repository implements a simple deep learning model for classifying handwritten digits from the MNIST dataset. The model is built using TensorFlow and Keras, adhering to the constraints of *Deep Learning with Python, 1st Edition* (DLWP) Chapters 1-4, specifically utilizing Dense and Dropout layers.

## Project Overview

This project aims to demonstrate a fundamental machine learning workflow, from data loading and preprocessing to model training and evaluation, using a classic dataset and basic neural network building blocks.  It is designed as a learning exercise following the principles outlined in DLWP.

## Repository Structure

```
mnist-digit-classifier-dlwp/
├── README.md                # This file
├── mnist_classifier.ipynb   # Jupyter notebook containing the code
├── data/                    # (Optional) Directory for any downloaded or preprocessed data
│   └── ...
└── models/                  # (Optional) Directory for saved model weights
    └── ...
```

*   **`README.md`:** Provides an overview of the project, instructions for setup, and a description of the repository structure.
*   **`mnist_classifier.ipynb`:**  The main Jupyter notebook containing the code for data loading, preprocessing, model building, training, and evaluation.  This notebook follows the DLWP 4.5 workflow.
*   **`data/`:** (Optional)  A directory to store any downloaded or preprocessed data.  This might include intermediate data files created during preprocessing, but is not strictly required since TensorFlow Datasets handles downloading the raw data.
*   **`models/`:** (Optional) A directory to save trained model weights (e.g., using `model.save_weights()`).

## Getting Started

1.  **Clone the repository:**

    ```bash
    git clone [repository URL]
    cd mnist-digit-classifier-dlwp
    ```

2.  **Install Dependencies:** (If necessary. If you're running this on colab or a pre-configured env, you might skip this)

    Ensure you have TensorFlow and TensorFlow Datasets installed.  You can install them using pip:

    ```bash
    pip install tensorflow tensorflow-datasets
    ```

3.  **Open the Jupyter Notebook:**

    Launch Jupyter Notebook and open `mnist_classifier.ipynb`.

    ```bash
    jupyter notebook
    ```

4.  **Follow the Notebook:**

    Execute the cells in the notebook sequentially, reading the explanations and observing the results.

## Implementation Details

The `mnist_classifier.ipynb` notebook implements the following steps, aligned with the DLWP workflow:

1.  **Problem Definition and Data Assembly:** Loading the MNIST dataset using `tensorflow-datasets`.
2.  **Choosing a Measure of Success:** Using accuracy as the evaluation metric.
3.  **Deciding on an Evaluation Protocol:** Using a separate test set provided by the dataset.
4.  **Data Preparation:**
    *   Reshaping the images into vectors.
    *   Scaling pixel values to the range \[0, 1].
5.  **Developing a Model That Does Better Than a Baseline:** Building a simple Dense network.
6.  **Scaling Up: Developing a Model That Overfits:** Increasing the capacity of the network.
7.  **Regularizing Your Model and Tuning Your Hyperparameters:** Adding Dropout layers to reduce overfitting.

## Code References

This project leverages code examples and concepts from *Deep Learning with Python, 1st Edition* (DLWP) by François Chollet. All code adapted from DLWP is explicitly referenced in the notebook.
