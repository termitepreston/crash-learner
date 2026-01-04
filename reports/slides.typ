#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *

#show: codly-init.with()
#codly(
  languages: (
    python: (name: "Python", icon: none, color: rgb("#3572A5")),
  ),
)

#let authors = (
  "Alazar Gebremehdin",
  "Hannibal Mussie",
  "Feruz Seid",
  "Yassin Bedru",
  "Samir Bahru",
)

#show math.equation: set text(font: "STIX Two Math")
#show raw: set text(font: "CMU Typewriter Text")

#set text(
  lang: "en",
  font: ("TeX Gyre Schola", "Noto Serif Ethiopic"),
  tracking: -0.03em,
  ligatures: true,
  number-type: "old-style",
)

#set par(justify: true)

#show: university-theme.with(
  aspect-ratio: "16-9",
  footer-a: none,
  config-info(
    title: [Classifying Car Crashes Using Neural Networks],
    subtitle: [_From Raw Data to Severity Prediction_],
    author: authors.join(", "),
    date: datetime.today(),
    instituition: [HiLCoE School of Computer Science & Technology],
  ),
)

#title-slide()

== Outline <touying:hidden>

#components.adaptive-columns(outline(title: none, indent: 1em, depth: 1))

= Introduction

== Objective & Overview

#slide[
  *Goal:* Predict the severity of road traffic accidents (*Fatal, Serious, Minor, PDO*) based on accident characteristics.

  *The Workflow:*
  1. *Data Understanding:* Handling massive missing data and inconsistencies.
  2. *Preparation:* Cleaning, Imputation, and Feature Engineering.
  3. *Modeling:* Designing a Multi-Layer Perceptron (MLP).
  4. *Training:* Managing class imbalance and overfitting.
  5. *Evaluation:* F1-Scores and Confusion Matrices.
]

= Data Understanding (EDA)

== Data Quality Challenges

#slide[
  *Initial Inspection:*
  - The raw dataset contained over 30 columns but suffered from severe quality issues.
  - *Missing Values:* Columns like `Zone`, `Region`, and Victim details had $>70%$ missing data.
  - *Inconsistencies:* Typos (e.g., "Augest", "Privategg") and impossible values (Age $> 90$).

  *Action:*
  - Dropped columns with $>70%$ missingness.
  - Standardized categorical labels (e.g., mapping `P.D.O`, `pdo` $->$ `PDO`).
]

== Missing Values Analysis

#slide[
  #align(center)[
    #image("figures/missing_values_cleaned.svg", height: 85%)
  ]
]

== Class Imbalance

#slide[
  *The Critical Challenge:*
  The dataset is heavily skewed towards *Minor Injuries* (~63%). *Fatal* accidents represent only ~3%.

  #align(center)[
    #image("figures/target_distribution_cleaned.svg", height: 60%)
  ]
]

= Data Preparation

== Feature Engineering: Time

#slide[
  *Problem:* Time is cyclical. 23:00 is close to 00:00, but numerically (23 vs 0) they are far apart.

  *Solution:* We encoded time using Sine and Cosine transformations.

  ```python
  # Feature Engineering Code Snippet
  def feature_engineering(df):
      # Extract Hour
      df['Hour'] = df['Time'].apply(extract_hour)

      # Cyclical Encoding
      df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
      df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

      return df.drop(columns=['Time'])
  ```
]

== Preprocessing Pipeline

#slide[
  Before feeding data into the Neural Network:

  1. *Imputation:*
    - Numerical (Age, Experience) $->$ *Median*
    - Categorical (Road Surface, Light) $->$ *Mode*
  2. *Scaling:*
    - `StandardScaler` applied to numerical inputs to normalize variance.
  3. *Encoding:*
    - `OneHotEncoder` for categorical variables.
  4. *Splitting:*
    - Train (70%) / Validation (15%) / Test (15%).
]

= Model Design

== Architecture Summary

#slide[
  Based on the execution results:

  - *Input Shape:* 230 Features (High dimensionality due to One-Hot Encoding).
  - *Total Parameters:* 38,917 (Lightweight model).
  - *Trainable Params:* 38,533.

  *Layer Structure:*
  - Input (230)
  - Dense (128) $->$ BN $->$ ReLU $->$ Dropout
  - Dense (64) $->$ BN $->$ ReLU $->$ Dropout
  - Output (5 Classes)
]

== Implementation Code

#slide[
  ```python
  # Actual Model Summary Output
  Layer (type)                Output Shape              Param #
  =================================================================
  Input_Layer (InputLayer)    (None, 230)               0
  Hidden_Layer_1 (Dense)      (None, 128)               29,568
  Batch_Norm_1                (None, 128)               512
  Dropout_1 (Dropout)         (None, 128)               0
  Hidden_Layer_2 (Dense)      (None, 64)                8,256
  Output_Layer (Dense)        (None, 5)                 325
  =================================================================
  Total params: 38,917
  ```
]

#matrix-slide[
  #rotate(-90deg)[
    #image("figures/model_architecture.png")
  ]
]


= Training & Evaluation

== Training Performance

#slide[
  #align(center)[
    #image("figures/training_curves.svg", height: 85%)
  ]

  *Insight:* The model converges quickly. Validation accuracy peaks around *98%*, indicating robust learning without significant overfitting.
]

== Confusion Matrix Results

#slide[
  #align(center)[
    #image("figures/confusion_matrix.svg", height: 85%)
  ]
]

== Analysis of Results

#slide[
  *Exceptional Performance:*
  - *Fatal Class:* 99% Recall (265 Correct, 2 Missed).
  - *Minor Class:* 98% Recall.

  *Critical Analysis (The "Why"):*
  - The high accuracy suggests the model effectively utilized casualty count features (e.g., `Number of fatalities`) present in the dataset.
  - While excellent for *classifying* historical records, this indicates that accident outcomes (casualties) are the strongest predictors of the severity label.
]

= Conclusion

== Summary

#slide[
  1. *Data Quality:* Cleaning and encoding resulted in 230 clean input features.
  2. *Model:* A 38k parameter MLP was sufficient to capture the relationships.
  3. *Results:* The model achieved ~98% test accuracy.

  *Recommendation:*
  - For future *predictive* systems (pre-accident), we recommend retraining the model *excluding* the `Number of casualties` columns to test predictive power based solely on environmental factors (Road type, Weather, etc.).
]

#slide[
  #align(center + horizon)[
    #text(size: 2em)[Thank You!]

    #v(1em)
    *Questions?*
  ]
]
