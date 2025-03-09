**DLWP 4.5: Universal Machine Learning Workflow**

**1. Problem Definition & Data Assembly**

*   **Define the Problem:**
    *   What is the input data (X)?
    *   What are you trying to predict (Y)?
    *   What type of problem is it? (Binary/Multiclass Classification, Regression, etc.)
*   **Assemble Dataset:**
    *   Ensure sufficient data is available for the chosen problem type.
*   **Hypotheses:**
    *   Outputs (Y) can be predicted from inputs (X).
    *   Available data is informative enough to learn the relationship between X and Y.
*   **Considerations:**
    *   **Non-Stationary Problems:** Model retraining with recent data or inputting the time variable.
    *   ML only memorizes seen patterns. Assumes the future behaves like the past.

**2. Choosing a Measure of Success**

*   Define what "success" means for your specific problem (accuracy, precision/recall, etc.).
*   Select a metric aligned with higher-level goals.
*   Examples:
    *   Balanced Classification: Accuracy, ROC AUC
    *   Imbalanced Classification: Precision, Recall
    *   Ranking/Multilabel: Mean Average Precision

**3. Deciding on an Evaluation Protocol**

*   Choose a method to measure progress:
    *   **Hold-out Validation Set:** Plenty of data.
    *   **K-Fold Cross-Validation:** Limited data.
    *   **Iterated K-Fold Validation:** High accuracy needed with limited data.

**4. Data Preparation**

*   Format data as tensors.
*   Scale values to a small range (e.g., [-1, 1] or [0, 1]).
*   Normalize heterogeneous data (features with different ranges).
*   Consider feature engineering (especially for small datasets).

**5. Developing a Model That Beats a Baseline**

*   Goal: Achieve "statistical power" (better than a random baseline).
*   If a baseline cannot be beaten, re-evaluate data or problem.
*   Key choices for initial model:
    *   **Last-Layer Activation:** Constrains output (e.g., sigmoid for binary classification).
    *   **Loss Function:** Matches problem type (e.g., binary_crossentropy).
    *   **Optimization Configuration:** Optimizer (e.g., rmsprop) and learning rate.
*   **Note:** May need to use proxy metric (such as crossentropy) instead of the success metric.

**6. Scaling Up: Develop an Overfitting Model**

*   Add layers.
*   Increase layer size.
*   Train for more epochs.
*   Monitor training/validation loss and metrics. Overfitting occurs when validation performance degrades.

**7. Regularizing & Hyperparameter Tuning**

*   Iteratively modify, train, and evaluate on the *validation* set (not test).
*   Techniques:
    *   Dropout
    *   Architecture changes (add/remove layers)
    *   L1/L2 regularization
    *   Hyperparameter tuning (learning rate, units per layer)
    *   Feature engineering (add/remove features)
*   **Caution:** Avoid overfitting to the validation data through excessive tuning.

**8. Final Evaluation**

*   Train the final model on *all* available data (training + validation).
*   Evaluate *once* on the test set.
*   If test performance is significantly worse than validation, the validation procedure was unreliable or there was overfitting to the validation data. Switch to a more reliable evaluation protocol.

**Chapter Summary**

*   Define problem and data.
*   Choose a success metric.
*   Determine an evaluation protocol.
*   Develop a model that beats a basic baseline.
*   Develop a model that overfits.
*   Regularize and tune hyperparameters.
