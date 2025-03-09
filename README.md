# CM3015 Machine Learning and Neural Networks - End of Term Coursework

## Overview

This project implements the deep learning workflow described in DLWP 4.5 (1st edition) to train a model on the MNIST dataset. The goal is to create a well-structured and documented report in a Jupyter notebook demonstrating a systematic approach to model development, experimentation, and interpretation of results.  The final submission is an HTML export of the Jupyter notebook.

## Dataset

*   **Name:** MNIST
*   **Description:** A dataset of handwritten digits (0-9).  Commonly used as a starting point for image classification tasks.

## Objective

To build and train a model that accurately classifies handwritten digits from the MNIST dataset using only the layers and techniques covered in DLWP Part 1 (Chapters 1-4).

## Implementation Details

1.  **DLWP 4.5 Workflow:** The project strictly adheres to the workflow outlined in DLWP 4.5:

    *   **Problem Definition & Data Assembly:** Define the task (image classification), load and preprocess the MNIST dataset.
    *   **Choosing a Measure of Success:** Select appropriate metrics for multiclass classification (e.g., accuracy, categorical cross-entropy).
    *   **Deciding on an Evaluation Protocol:** Implement a suitable evaluation strategy (e.g., hold-out validation set).
    *   **Data Preparation:**  Reshape and scale the MNIST data (e.g., normalize pixel values).
    *   **Developing a Model That Beats a Baseline:**  Create a basic model with Dense and Dropout layers and demonstrate that it outperforms random chance.
    *   **Scaling Up: Develop an Overfitting Model:** Add layers and increase the number of units to overfit the training data.
    *   **Regularizing & Hyperparameter Tuning:** Apply regularization techniques (Dropout, L1/L2 regularization) and tune hyperparameters to improve generalization performance on the validation set.
    *   **Final Evaluation:** Train the final model on the combined training and validation data and evaluate its performance on the test set.

2.  **Technology:**

    *   **Programming Language:** Python
    *   **Deep Learning Framework:** TensorFlow (Keras API)
    *   **Development Environment:** Jupyter Notebook

3.  **Model Architecture:**

    *   Limited to `tensorflow.keras.layers.Dense` and `tensorflow.keras.layers.Dropout` layers, as per DLWP Part 1.
    *   Experiment with different numbers of layers, units per layer, and dropout rates.

4.  **Code Structure:**

    *   Modular code for data loading, preprocessing, model building, training, evaluation, and visualization.

## Submission

*   **File:** HTML export of the Jupyter notebook (`.html`)
*   **Content:** The notebook should contain:
    *   Well-structured markdown sections with headings, subheadings, and tables to describe each step of the DLWP workflow.
    *   Clear explanations of the code, experiments, and results.
    *   Visualizations (e.g., training curves, example predictions) to support the analysis.
*   **DO NOT SUBMIT:**
    *   The original Jupyter Notebook file (`.ipynb`)
    *   Any data files (e.g., MNIST dataset files)

## Evaluation Criteria

*   **Report Quality:**  Structure, clarity, and readability of the Jupyter notebook as a report.
*   **Adherence to DLWP Workflow:**  Complete and correct implementation of the steps in DLWP 4.5.
*   **Systematic Investigation:**  Demonstrated exploration of different model architectures, hyperparameters, and regularization techniques.
*   **Interpretation of Results:**  Clear and insightful analysis of the training and validation performance, including identification of overfitting and strategies for improvement.
*   **Modular Programming:** Use of functions and classes to organize the code.
*   **Extensive Experimentation:**  A variety of experiments with different model configurations and hyperparameters.
*   **Understanding and Technique:** Demonstrated understanding of deep learning concepts and techniques.

## Referencing

*   All code that is not original must be properly referenced, including code from DLWP, the video notebooks, and any other sources.  Credit will be given for model assembly using third-party code.

## Colab Users

*   If using Google Colab, download the notebook and load it into Jupyter locally to export as HTML, or follow these steps:

    1.  Download the `.ipynb` file from Colab.
    2.  Upload the `.ipynb` file back into Colab's session storage (drag and drop into the file browser on the left).
    3.  Run the following script in a Colab cell:

        ```python
        %%shell
        jupyter nbconvert --to html /content/your_notebook_name.ipynb
        ```

        Replace `/content/your_notebook_name.ipynb` with the correct path to your notebook in Colab's file system.
    4.  Download the generated HTML file.
