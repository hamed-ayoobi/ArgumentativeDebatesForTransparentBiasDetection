# Argumentative Debates for Transparent Bias Detection

This repository contains the code to reproduce the results presented in the AAAI 2026 paper *"Argumentative Debates for Transparent Bias Detection"*. The proposed method offers a transparent, argumentation-based framework for detecting algorithmic bias across individuals and groups, using Quantitative Bipolar Argumentation Frameworks (QBAFs).

## 📄 Paper

* Main paper
* Appendix (Supplementary Material)

## ⚙️ Installation

Ensure Python 3.11.10 is installed.

```bash
pip install -r requirements.txt
```

## 🧪 Running the Experiments

### 🧠 Synthetic Bias Detection (Tables 1 & 5)

These experiments involve hand-crafted classifiers (Global 1, Global 2, Local 1) designed to encode synthetic biases.

* Script:
  `BiasDetection(SingleNeighborhood).py`

* Set the dataset inside the script:

  ```python
  dataset_name = datasets.SyntheticAdult
  ```

* Results appear in:

  * **Table 1** (main paper)
  * **Table 5** (appendix)

---

### 📊 Aggregated Bias Detection (Table 2)

This experiment evaluates synthetic models using multiple neighbourhoods of varying sizes.

* Script:
  `BiasDetection(MultipleNeighborhoodsAggregation).py`

* Set the dataset:

  ```python
  dataset_name = datasets.SyntheticAdult
  ```

* Results appear in:

  * **Table 2**

---

### 🌍 Trained Classifier Bias Detection (Tables 3, 6, 7)

These experiments evaluate real-world trained classifiers on COMPAS and Bank datasets.

* Script:
  `BiasDetection(MultipleNeighborhoodsAggregation).py`

* Set the dataset:

  ```python
  dataset_name = datasets.COMPAS  # for Table 3a, Table 6
  dataset_name = datasets.Bank    # for Table 3b, Table 7
  ```

* Results appear in:

  * **Table 3a** (COMPAS), **Table 3b** (Bank)
  * **Tables 6 & 7** (Appendix – example cases)

---

### 🤖 LLM Bias Detection (Table 4)

Evaluates bias in large language models (e.g., ChatGPT-4o) acting as classifiers.

* Script:
  `LLM_experiments.py`

* Set the dataset:

  ```python
  dataset_name = datasets.LLM_COMPAS
  dataset_name = datasets.LLM_Bank
  ```

* Results appear in:

  * **Table 4**

---

### 📊  Sensitivity Analysis

To run sensitivity analysis in cross-neighbourhood model, run the SensitivityAnalysis.py script. 

---

## 🧾 Notes

* All experiments can be run on CPU (no GPU required).
* Synthetic model definitions and bias injection logic are located in `datasets.py`.
* **Quadratic Energy semantics** is used by default for evaluating argument strength in QBAFs.

