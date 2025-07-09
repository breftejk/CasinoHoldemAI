import os
import subprocess

# Directory for LaTeX documentation
docs_dir = "docs"
os.makedirs(docs_dir, exist_ok=True)

# LaTeX section files with academic-style content
sections = {
    "introduction.tex": r"""
\section{Introduction}
CasinoHoldemAI is a production-grade decision engine for Casino Hold'em, integrating rigorous Monte Carlo simulation with a gradient-boosted decision tree classifier (XGBoost). Key contributions include:
\begin{itemize}
  \item A configurable Monte Carlo equity estimator with iteration count $N$.
  \item Rich feature extraction combining combinatorial and probabilistic metrics.
  \item A robust XGBoost training pipeline for optimistic CALL/FOLD classification.
\end{itemize}
""",
    "installation.tex": r"""
\section{Installation}
\begin{enumerate}
  \item Create a virtual environment:
  \begin{verbatim}
python3 -m venv .venv
source .venv/bin/activate
  \end{verbatim}
  \item Upgrade core tools:
  \begin{verbatim}
pip install --upgrade pip setuptools wheel Cython
  \end{verbatim}
  \item Install dependencies and package:
  \begin{verbatim}
pip install numpy pandas eval7 scikit-learn xgboost joblib tqdm
pip install .
  \end{verbatim}
  \item Verify:
  \begin{verbatim}
casino-ai --help
  \end{verbatim}
\end{enumerate}
""",
    "theory.tex": r"""
\section{Theoretical Foundations}

\subsection{Monte Carlo Simulation}
Let $W_i$ be the indicator of a win on trial $i$. Then:
\[
\hat{p} = \frac{1}{N} \sum_{i=1}^N W_i, 
\quad \mathrm{Var}(\hat{p}) = \frac{\hat{p}(1-\hat{p})}{N}.
\]
By the Central Limit Theorem, for large $N$:
\[
\hat{p} \sim \mathcal{N}\!\Bigl(p,\,\frac{p(1-p)}{N}\Bigr).
\]
A $100(1-\alpha)\%$ confidence interval:
\[
\hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{N}}.
\]

\subsection{Feature Engineering}
Define rank frequency vector $\mathbf{c} = [c_2,\dots,c_{\mathrm{A}}]$. Pattern indicators:
\[
\text{Pair} = \mathbb{I}\bigl(\max(\mathbf{c})\ge2\bigr),\quad
\text{Trips} = \mathbb{I}\bigl(\max(\mathbf{c})\ge3\bigr).
\]
Expected outs calculation:
\[
E[\text{outs}] = \sum_k o_k \frac{\binom{o_k}{1}\binom{R - o_k}{T-1}}{\binom{R}{T}},
\]
normalized to $[0,1]$.
""",
    "architecture.tex": r"""
\section{System Architecture}
\begin{description}
  \item[CLI:] \texttt{argparse}-based command parsing.
  \item[Simulator:] $\mathcal{O}(N)$ Monte Carlo per hand, parallel via \texttt{joblib.Parallel}.
  \item[FeatureExtractor:] Computation of combinatorial/statistical features.
  \item[DataGenerator:] Batch orchestration with progress via \texttt{tqdm}.
  \item[ModelTrainer:] XGBoost training with train/validation split.
  \item[PokerAI:] Real-time simulation + ML inference.
\end{description}
""",
    "ml_details.tex": r"""
\section{Machine Learning Details}

\subsection{XGBoost Ensemble}
Model ensemble:
\[
F(x) = \sum_{m=1}^M f_m(x),\quad
\hat{y} = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}}.
\]
Objective:
\[
\mathcal{L} = \sum_i \ell(y_i,\hat{y}_i) + \sum_{m}\Omega(f_m),
\quad
\Omega(f) = \gamma T + \tfrac12\lambda\lVert w\rVert^2.
\]

\subsection{Hyperparameter Tuning}
Grid search over:
\[
\{\text{max\_depth}, \eta, \text{subsample}, \text{colsample\_bytree}, \lambda, \gamma\}
\]
using 5-fold CV to minimize logloss.

\subsection{Bias-Variance Tradeoff}
Generalization error:
\[
\mathrm{Err} = \mathrm{Bias}^2 + \mathrm{Variance} + \text{Noise}.
\]
Regularization and learning rate control complexity.
""",
    "features.tex": r"""
\section{Data \& Feature Engineering}

\subsection{Monte Carlo Outputs}
\begin{itemize}
  \item \texttt{win\_rate}, \texttt{tie\_rate}
  \item 95\% CI: $\hat{p}\pm1.96\sqrt{\hat{p}(1-\hat{p})/N}$
\end{itemize}

\subsection{Hand Patterns}
Indicator vector length 9: pair, two\_pair, trips, straight, flush, full\_house, quads, straight\_flush, royal\_flush.

\subsection{Card Encoding}
Ranks $\{2,\dots,9,T,J,Q,K,A\}\to\{2,\dots,14\}$; suits mapped to $\{1,\dots,4\}$.
""",
    "api.tex": r"""
\section{API Reference}

\subsection{Generate Data (\texttt{gen})}
\begin{verbatim}
casino-ai gen --n N --out PATH [--iters I] [--workers W]
\end{verbatim}

\subsection{Train Model (\texttt{train})}
\begin{verbatim}
casino-ai train --in PATH --model PATH
\end{verbatim}

\subsection{Predict Decision (\texttt{pred})}
\begin{verbatim}
casino-ai pred --model PATH --cards C1,C2 --board B1,B2,B3 [--threshold T]
\end{verbatim}
""",
    "disclaimer.tex": r"""
\section{Disclaimer}

This software is provided \textit{as-is}, without any express or implied warranty. The author assumes no liability for decisions made using this tool.
""",
}

# Write each LaTeX section
for filename, content in sections.items():
    path = os.path.join(docs_dir, filename)
    with open(path, "w") as f:
        f.write(content.strip() + "\n")

# Master LaTeX document with necessary packages
master = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{geometry}
\geometry{margin=1in}

\title{CasinoHoldemAI Documentation}
\author{Marcin Kondrat (\texttt{@breftejk})}
\date{\today}

\begin{document}
\maketitle
\input{introduction.tex}
\input{installation.tex}
\input{theory.tex}
\input{architecture.tex}
\input{ml_details.tex}
\input{features.tex}
\input{api.tex}
\input{disclaimer.tex}
\end{document}
"""
with open(os.path.join(docs_dir, "documentation.tex"), "w") as f:
    f.write(master.strip() + "\n")

# Compile to PDF if available
try:
    subprocess.run(["pdflatex", "-output-directory", docs_dir, os.path.join(docs_dir, "documentation.tex")], check=True)
    print("Academic-grade PDF generated at docs/documentation.pdf")
except FileNotFoundError:
    print("pdflatex not found; LaTeX source prepared at docs/documentation.tex")
