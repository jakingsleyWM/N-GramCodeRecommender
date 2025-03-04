* Introduction  
* Getting Started  
  * Preparations  
  * Install Packages  
  * Run N-Gram  
* Report

**Introduction**

Code completion is a crucial feature in software development, assisting developers by predicting and suggesting the next tokens in an incomplete code snippet. The N-gram model predicts the next token in a sequence by learning the probabilities of token occurrences n-tokens before the next. In this project, I implement an N-gram probabilistic language model for Java code completion. 

**Getting Started**

This project was implemented using Python 3.11, with compatibility for macOS and Linux systems. Windows compatibility is not guaranteed. 

**Preparations:**

1) Clone the repository to your workspace

\~ $ git clone https://github.com/jakingsleyWM/N-GramCodeRecommender

2) Navigate into repository

\~ $ cd N-GramCodeRecommender

3) Set up virtual environment and activate it

**For macOS/Linux:**

\~/N-GramCodeRecommender$ python \-m venv ./venv/

\~/N-GramCodeRecommender$ source venv/bin/activate

(venv) \~/N-GramCodeRecommender$ 

**Install Packages:** 

1) Install the required dependencies:

(venv) \~/N-GramCodeRecommender$ pip install \-r requirements.txt

**Run N-Gram:** 

The script takes a corpus of Java methods as input and automatically identifies the best-performing model based on a specific N-value. It then evaluates the selected model on the test set extracted according to the assignment specifications. Since the training corpus differs from both the instructor-provided dataset and my own dataset, I store the results in a file named results\_provided\_model.json to distinguish them accordingly.

(venv) \~/N-GramCodeRecommender$ python CodeRecommender.py teacher\_data.txt

**Report**

The assignment report is available in the file AssignmentReport.pdf .