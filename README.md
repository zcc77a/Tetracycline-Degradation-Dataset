# Gibbs Free Energy Changes ($\Delta G$) of Tetracycline Degradation Dataset

[![Institution](https://img.shields.io/badge/Institution-Nanjing%20University-blue)](https://www.nju.edu.cn/)

## 📖 Overview

This repository hosts a mechanistic reaction database constructed for the analysis of transformation energetics in the **photocatalytic degradation of tetracycline** and related compounds. The dataset encompasses **120 overall stoichiometrically balanced reactions** decomposed into **9,533 elementary reaction steps**.

For each reaction step, high-fidelity thermodynamic data ($\Delta G$) and comprehensive reaction descriptors are compiled. This dataset is designed to facilitate machine learning (ML) analysis, thermodynamic modeling, and reaction pathway discovery in environmental chemistry and photocatalysis.



## 📂 Dataset Contents

The dataset includes detailed entries for each elementary reaction step, structured to support data-driven research.

### 1. Reaction Scope

* **Total Reactions:** 120 overall reactions.
* **Elementary Steps:** 9,533 specific steps.
* **Chemical Species Involved:**
  * **Target Transformation Products:** $TP_i$ (Reactant), $TP_j$ (Product).
  * **Environmental Species:** $O_2$, $H_2O$, $CO_2$, $NH_4^+$, $NO_3^-$.
  * **Reactive Oxygen Species (ROS):** Singlet oxygen ($^1O_2$), Superoxide ($O_2^{\bullet-}$), Hydroxyl radical ($^\bullet OH$).
  * **Photogenerated Charge Carriers:** Electrons ($e^-$), Holes ($h^+$).

### 2. Data Features (Descriptors)

Each entry in the dataset is characterized by the following features:

#### A. Stoichiometric Descriptors

Stoichiometric coefficients are signed to indicate reactants ($x < 0$) and products ($y > 0$). Fractional values (e.g., $0.5 O_2$) are used to strictly preserve thermodynamic consistency.

* **$x$**: Stoichiometric coefficients of reactants.
* **$y$**: Stoichiometric coefficients of products.

#### B. Quantum Chemical Features

Key electronic properties of the transformation products are included:

* **$E_{HOMO}$**: Energy of the Highest Occupied Molecular Orbital for reactants ($TP_i$) and products ($TP_j$).
* **$\Delta E_{H-L}$**: HOMO-LUMO gap energy for reactants ($TP_i$) and products ($TP_j$).

#### C. Thermodynamic Data (Target Variable)

* **$\Delta G$**: The standard Gibbs free energy change for the reaction step.

## 🧪 Computational Methodology

All thermodynamic data were computed using Density Functional Theory (DFT) with the following specifications:

* **Functional:** M06-2X
* **Basis Set:** 6-311+G(d,p)
* **Solvation Model:** SMD (Solvation Model based on Density)
* **Conditions:** 298.15 K, 1 atm

These calculations ensure high accuracy for describing the thermodynamics of organic degradation in aqueous environments.
