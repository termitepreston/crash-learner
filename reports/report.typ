#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#import "@preview/lovelace:0.3.0": *
#show: codly-init.with()


#let hilcoe-report(
  members: (),
  title: "",
  course-name: "Artificial Intelligence",
  course-code: "CS488",
  batch: "DRB2202",
  section: "A",
  instructor: "",
  term: "Autumn 2025",
  date: datetime.today(),
  show-outline: false,
  body,
) = {
  set page(
    paper: "a4",
    numbering: "1.",
    number-align: center,
    footer: context {
      let page-number = counter(page).get().at(0)
      if page-number > 1 {
        line(length: 100%, stroke: 0.5pt)
        v(-2pt)
        text(size: 12pt, weight: "regular")[
          HiLCoE
          #h(1fr)
          #page-number
          #h(1fr)
          2025
        ]
      }
    },
  )

  set text(
    size: 10pt,
    font: "STIX Two Text",
  )

  show math.equation: set text(font: "STIX Two Math")
  show raw: set text(font: "CMU Typewriter Text")

  set par(justify: true)


  set heading(numbering: "1.")

  align(
    figure(image("images/hilcoe-logo.svg", height: 120pt)),
  )
  // title page
  align(
    center,
    par(spacing: 30pt, text(size: 48pt, tracking: -0.03em, font: "Cooper", [HiLCoE])),
  )

  align(
    center,
    text(size: 18pt, font: "CMU Sans Serif", weight: "medium")[School of Computer Science \ & Technology],
  )

  v(30pt)

  align(center, par(
    leading: 1.2em,
    text(
      size: 40pt,
      font: "CMU Sans Serif",
      tracking: -0.02em,
    )[#emph(title)],
  ))

  v(40pt)


  // meta:
  //
  //
  //
  members = members.sorted()

  let ct = body => text(font: "Inter Display", size: 10pt)[
    #body
  ]


  align(
    center,
    block(
      fill: rgb("#ececfaaa"),
      stroke: 0.5pt,
      inset: (
        right: 36pt,
        left: 36pt,
        top: 20pt,
        bottom: 20pt,
      ),
      table(
        columns: (auto, auto),
        align: (right, left),
        inset: 8pt,
        stroke: none,
        ct()[*Members*],
        for member in members [
          #ct(member) \
        ],

        ct()[*Course*], ct()[#course-name (#course-code)],
        ct()[*Instructor*], ct()[#instructor],
        ct()[*Batch/Section*], ct()[#batch/#section],
        ct()[*Term*], ct()[#term],
        ct()[*Date*], ct()[#date.display("[month repr:short] [day], [year]")],
      ),
    ),
  )

  pagebreak()

  if show-outline {
    outline()

    pagebreak()
  }


  body
}


#show: hilcoe-report.with(
  title: [Group Report \ on \ Machine Learning],
  instructor: "Dr. Seyoum Abebe",
  members: (
    "Alazar Gebremehdin",
    "Hannibal Mussie",
    "Feruz Seid",
    "Yassin Bedru",
    "Samir Bahru",
  ),
)


= Introduction

The objective of this project is to build, train, and evaluate a Neural Network classification model to predict the severity of road traffic accidents based on a dataset from Addis Ababa. The target variable, `Accident Type`, classifies accidents into four categories: *Fatal*, *Serious Injury*, *Minor Injury*, and *Property Damage Only (PDO)*.

This report summarizes the end-to-end workflow, from Exploratory Data Analysis (EDA) and rigorous data cleaning to model architecture design and final performance evaluation.

= Data Understanding and Preparation

== Exploratory Data Analysis (EDA)
Initial analysis revealed significant data quality challenges. The raw dataset contained over 30 features, but many suffered from high rates of missing values.
- *Missing Data:* Columns such as `Zone`, `Region`, and specific victim details (e.g., `Victim-2 Movement`) had missing rates exceeding 70%.
- *Class Imbalance:* The target variable was heavily skewed. *Minor Injuries* accounted for approximately 63% of the data, while *Fatal* accidents represented only ~3%.

#figure(
  image("figures/target_distribution_cleaned.svg", width: 80%),
  caption: [Distribution of Accident Severity (Target Variable). Note the severe imbalance.],
)

== Data Cleaning & Standardization
Before modeling, a custom cleaning pipeline was implemented:
1. *Standardization:* Typos were corrected (e.g., "Augest" $->$ "August", "Privategg" $->$ "Private"). Amharic terms like "Amet" (Year) and "Wor" (Month) were standardized.
2. *Outlier Removal:* Impossible values (e.g., Driver Age > 90, Experience > 60 years) were treated as missing.
3. *Feature Dropping:* Columns with $>70%$ missing data were removed to reduce noise.

== Feature Engineering & Preprocessing
To prepare the data for the Neural Network:
- *Cyclical Encoding:* The time of day was converted from linear hours (0-23) into sine and cosine components to preserve the temporal proximity between 23:00 and 00:00.
- *Imputation:* Median imputation was used for numerical features (e.g., Age) and Mode imputation for categorical features.
- *Scaling & Encoding:* Numerical features were standardized using `StandardScaler`. Categorical variables were transformed using `OneHotEncoding`.

= Model Design

== Architecture
A Multi-Layer Perceptron (MLP) was designed using TensorFlow/Keras. The architecture features a "funnel" design to compress high-dimensional inputs into abstract representations.

- *Input Layer:* ~100 features (after One-Hot Encoding).
- *Hidden Layer 1:* 128 Neurons, ReLU activation, Batch Normalization.
- *Hidden Layer 2:* 64 Neurons, ReLU activation, Batch Normalization.
- *Output Layer:* 4 Neurons with Softmax activation (representing probabilities for the 4 classes).

#figure(
  image("figures/model_architecture.png", width: 40%),
  caption: [Neural Network Architecture Diagram],
)

== Design Justification
- *ReLU Activation:* Selected to prevent the vanishing gradient problem.
- *Dropout (0.3 - 0.4):* Applied after hidden layers to randomly deactivate neurons during training, forcing the network to learn robust features and preventing overfitting.
- *L2 Regularization:* Added to penalize large weights, further reducing model complexity.

= Training Process

The model was trained for up to 100 epochs using the *Adam* optimizer. To address the class imbalance identified in EDA, *Class Weights* were computed and applied to the Loss Function. This penalizes the model more heavily for misclassifying rare classes (Fatal/Serious) than common ones (Minor).

*Overfitting Mitigation:*
- *Early Stopping:* Training halted automatically when Validation Loss failed to improve for 15 epochs.
- *Model Checkpointing:* Only the model version with the highest Validation Accuracy was saved.

#figure(
  image("figures/training_curves.svg", width: 100%),
  caption: [Training and Validation Performance Curves. Note the convergence point where Early Stopping triggers.],
)

= Evaluation and Interpretation

== Performance Metrics
The model was evaluated on an unseen Test Set (15% of data).

#figure(
  image("figures/confusion_matrix.svg", width: 100%),
  caption: [Confusion Matrix (Counts and Normalized). Ideally, the diagonal elements should be high.],
)

== Analysis of Results
1. *Strengths:*
   - The model achieves high recall on the majority class (*Minor Injury*), effectively identifying the most common accident types.
   - The use of class weights allowed the model to learn patterns for *Serious* and *Fatal* accidents better than a baseline model, which would have ignored them entirely.

2. *Weaknesses:*
   - *Precision on Minority Classes:* Due to the extreme rarity of Fatal accidents (only ~60 cases in the raw data), the model likely struggles to distinguish them perfectly from Serious injuries. This results in some misclassification between adjacent severity levels.
   - *Data Noise:* Despite cleaning, the inherent randomness in accident data (e.g., weather conditions labeled subjectively) limits the upper bound of accuracy.

= Recommendations & Conclusion

== Potential Improvements
1. *Advanced Architectures:* Implementing *TabNet* or *Wide & Deep* networks could better capture the interactions between categorical features (Road Type, Junction) and numerical ones.
2. *Resampling Strategies:* Instead of just class weights, applying *SMOTE (Synthetic Minority Over-sampling Technique)* to generate synthetic examples of Fatal accidents during training could improve recall.
3. *Feature Selection:* Using Recursive Feature Elimination (RFE) to remove irrelevant categorical features (e.g., specific rare "Kebele" or "Sub-city" values) could reduce noise.

== Reflection on Workflow
The project demonstrated that *Data Preparation* is the most critical phase. The raw data required extensive cleaning before any modeling could succeed. The transition from a raw, noisy Excel sheet to a structured Neural Network pipeline highlights the importance of robust feature engineering (like cyclical time encoding) and rigorous evaluation beyond simple accuracy.
