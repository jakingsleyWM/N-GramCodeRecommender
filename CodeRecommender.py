import pandas as pd
from pydriller import Repository
import os
import re
import javalang
from javalang.parse import parse
from javalang.tree import MethodDeclaration
import git
import csv
import random
import json
import numpy as np
from collections import defaultdict, Counter
from pygments.lexers.jvm import JavaLexer
from pygments.token import Token
from pygments.lexers import get_lexer_by_name
import argparse

# Define functions for processing repositories and extracting methods
def extract_methods_from_java(code):
    """
    Extract methods from Java source code using javalang parser.

    Args:
        code (str): The Java source code.

    Returns:
        list: A list of tuples containing method names and their full source code.
    """
    methods = []
    try:
        # Parse the code into an Abstract Syntax Tree (AST)
        tree = javalang.parse.parse(code)
        lines = code.splitlines()

        # Traverse the tree to find method declarations
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            method_name = node.name

            # Determine the start and end lines of the method
            start_line = node.position.line - 1
            end_line = None

            # Use the body of the method to determine its end position
            if node.body:
                last_statement = node.body[-1]
                if hasattr(last_statement, 'position') and last_statement.position:
                    end_line = last_statement.position.line

            # Extract method code
            if end_line:
                method_code = "\n".join(lines[start_line:end_line+1])
            else:
                # If end_line couldn't be determined, extract up to the end of the file
                method_code = "\n".join(lines[start_line:])

            methods.append((method_name, method_code))

    except Exception as e:
        print(f"Error parsing Java code: {e}")
    return methods


def extract_methods_to_csv_from_master(repo_path, output_csv):
    """
    Extract methods from Java files in the master/main branch and save them in a CSV file.

    Args:
        repo_path (str): Path to the Git repository.
        output_csv (str): Path to the output CSV file.
    """
    repo_name = repo_path.split('/')[-1]
    local_repo_path = os.path.join(os.getcwd(), repo_name)

    if not os.path.exists(local_repo_path):
        git.Repo.clone_from(repo_path, local_repo_path)

    try:
        repo = git.Repo(local_repo_path)
        branch_name = repo.active_branch.name
    except git.exc.InvalidGitRepositoryError:
        branch_name = "main"

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Commit Hash", "File Name", "Method Name", "Method Code", "Commit Link"])

        for commit in Repository(repo_path, only_in_branch=branch_name).traverse_commits():
            print(f"Processing commit: {commit.hash}")

            for modified_file in commit.modified_files:
                if modified_file.filename.endswith(".java") and modified_file.source_code:
                    methods = extract_methods_from_java(modified_file.source_code)

                    for method_name, method_code in methods:
                        commit_link = f"{repo_path}/commit/{commit.hash}"
                        csv_writer.writerow([commit.hash, modified_file.filename, method_name, method_code, commit_link])

                    print(f"Extracted methods from {modified_file.filename} in commit {commit.hash}")


def extract_methods_to_csv(repo_path, output_csv):
    """
    Extract methods from Java files in a repository and save them in a CSV file.

    Args:
        repo_path (str): Path to the Git repository.
        output_csv (str): Path to the output CSV file.
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Branch Name", "Commit Hash", "File Name", "Method Name", "Method Code", "Commit Link"])

        branch_name = "master"
        for commit in Repository(repo_path, only_in_branch=branch_name).traverse_commits():
            print(f"Processing commit: {commit.hash}")

            for modified_file in commit.modified_files:
                if modified_file.filename.endswith(".java") and modified_file.source_code:
                    methods = extract_methods_from_java(modified_file.source_code)

                    for method_name, method_code in methods:
                        commit_link = f"{repo_path}/commit/{commit.hash}"
                        csv_writer.writerow([branch_name, commit.hash, modified_file.filename, method_name, method_code, commit_link])

                    print(f"Extracted methods from {modified_file.filename} in commit {commit.hash}")

# Function to remove duplicates based on method content
def remove_duplicates(data):
    return data.drop_duplicates(subset="Method Code", keep="first")

# Filter out methods that do not have ASCII characters
def filter_ascii_methods(data):
    data = data[data["Method Code"].apply(lambda x: all(ord(char) < 128 for char in x))]
    return data

# Remove outliers based on method length
def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    method_lengths = data["Method Code"].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile / 100)
    upper_bound = method_lengths.quantile(upper_percentile / 100)
    return data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]

# Remove boilerplate methods (getters/setters)
def remove_boilerplate_methods(data):
    boilerplate_patterns = [
        r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Setter methods
        r"\bget[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Getter methods
    ]
    boilerplate_regex = re.compile("|".join(boilerplate_patterns))
    data = data[~data["Method Code"].apply(lambda x: bool(boilerplate_regex.search(x)))]
    return data

# Function to remove comments from the code
def remove_comments_from_dataframe(df: pd.DataFrame, method_column: str, language: str) -> pd.DataFrame:
    def remove_comments(code):
        lexer = get_lexer_by_name(language)
        tokens = lexer.get_tokens(code)
        clean_code = ''.join(token[1] for token in tokens if not (lambda t: t[0] in Token.Comment)(token))
        return clean_code

    df["Method Code No Comments"] = df[method_column].apply(remove_comments)
    return df

def tokenize_java_code(code):
    """Tokenizes Java code and returns a list of meaningful tokens."""
    lexer = JavaLexer()
    tokens = [t[1] for t in lexer.get_tokens(code) if t[0] not in Token.Text and not t[1].isspace()]
    return tokens

# N-Gram model class definition
class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.total_counts = Counter()

    def train(self, tokenized_sequences):
        for tokens in tokenized_sequences:
            for i in range(len(tokens) - self.n + 1):
                prefix = tuple(tokens[i:i+self.n-1])
                next_token = tokens[i+self.n-1]
                self.ngram_counts[prefix][next_token] += 1
                self.total_counts[prefix] += 1

    def get_probability(self, prefix, token):
        prefix = tuple(prefix[-(self.n-1):])
        if prefix in self.ngram_counts:
            return self.ngram_counts[prefix][token] / self.total_counts[prefix]
        return 1e-10

    def predict(self, prefix):
        prefix = tuple(prefix[-(self.n-1):])
        if prefix in self.ngram_counts:
            return self.ngram_counts[prefix].most_common(1)[0]
        return (None, 1e-10)

# Function to calculate perplexity
def calculate_perplexity(model, tokenized_sequences):
    log_prob_sum = 0
    total_tokens = 0

    for tokens in tokenized_sequences:
        for i in range(len(tokens) - model.n + 1):
            prefix = tuple(tokens[i:i+model.n-1])
            next_token = tokens[i+model.n-1]
            prob = model.get_probability(prefix, next_token)

            log_prob_sum += np.log2(prob) if prob > 0 else np.log2(1e-10)
            total_tokens += 1

    return 2 ** (-log_prob_sum / total_tokens)

# Function to generate predictions
def generate_predictions(model, test_inputs):
    predictions = {}

    for idx, prefix in enumerate(test_inputs):
        prediction, prob = model.predict(prefix)
        predictions[str(idx)] = [(prediction, str(prob))] if prediction else [("", "0.0")]

    return predictions

# Argument parsing to accept teacher_data.txt file path
def parse_args():
    parser = argparse.ArgumentParser(description="Run N-Gram Code Recommender.")
    parser.add_argument("teacher_data_file", type=str, help="Path to the teacher data text file")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    df_res = pd.read_csv('small.csv')

    repoList = []
    for idx,row in df_res.iterrows():
        repoList.append("https://www.github.com/{}".format(row['name']))
        repoList[0:5]
    
    # Define a single output CSV file
    output_csv_file = "combined_methods.csv"

    # Initialize an empty DataFrame to store all methods
    all_methods = pd.DataFrame()
    x = 4  # Number of repositories to select
    random_repos = random.sample(repoList, x)  # Randomly select x repos

    for repo in random_repos:  # Adjust range as needed
        print(f"Processing repo: {repo}")

        # Extract methods into a temporary CSV file (or directly into a DataFrame)
        temp_output_csv = "temp_combined_methods.csv"
        extract_methods_to_csv_from_master(repo, temp_output_csv)

        # Read the extracted methods from this repo
        temp_data = pd.read_csv(temp_output_csv)

        # Append to the main DataFrame
        all_methods = pd.concat([all_methods, temp_data], ignore_index=True)

    # Save all collected methods into a single CSV file
    all_methods.to_csv(output_csv_file, index=False)
    print(f"All extracted methods saved to {output_csv_file}")

    data = pd.read_csv('combined_methods.csv')
    # Load teacher dataset
    with open(args.teacher_data_file, "r") as f:
        teacher_methods = [line.strip().split() for line in f.readlines()]  # Tokenized teacher dataset

    # Tokenize all Java methods
    tokenized_methods = data["Method Code"].apply(tokenize_java_code).tolist()

    # Split dataset for training, evaluation, and testing
    random.shuffle(tokenized_methods)
    split_train = int(0.8 * len(tokenized_methods))
    split_eval = int(0.9 * len(tokenized_methods))
    train_set, eval_set, test_set = tokenized_methods[:split_train], tokenized_methods[split_train:split_eval], tokenized_methods[split_eval:]

    # Determine best N using perplexity
    n_values = [3, 5, 9]
    perplexities = {}

    for n in n_values:
        model = NGramModel(n)
        model.train(train_set)
        perplexity = calculate_perplexity(model, eval_set)
        perplexities[n] = perplexity

    best_n = min(perplexities, key=perplexities.get)
    print(f"Best-performing n: {best_n}")
    print(f"Best-performing n eval-set perplexity: {perplexities[n]}")

    best_model = NGramModel(best_n)
    best_model.train(train_set)

    # Select 100 random test examples
    random_test_inputs = random.sample(test_set, 100)

    # Generate predictions for student model
    results_student = generate_predictions(best_model, random_test_inputs)

    perp = calculate_perplexity(best_model, test_set)
    print(f"Best-performing n test-set perplexity: {perp}")

    # Save student model results
    with open("results_student_model.json", "w") as f:
        json.dump(results_student, f, indent=4)

    # Repeat process for teacher model
    random.shuffle(teacher_methods)
    split_train = int(0.8 * len(teacher_methods))
    split_eval = int(0.9 * len(teacher_methods))
    train_set, eval_set, test_set = teacher_methods[:split_train], teacher_methods[split_train:split_eval], teacher_methods[split_eval:]

    perplexities_teacher = {}

    for n in n_values:
        model_teacher = NGramModel(n)
        model_teacher.train(train_set)
        perplexity = calculate_perplexity(model_teacher, eval_set)
        perplexities_teacher[n] = perplexity

    best_n_teacher = min(perplexities_teacher, key=perplexities_teacher.get)
    print(f"Best-performing n for teacher: {best_n_teacher}")
    print(f"Best-performing n eval-set perplexity for teacher: {perplexities_teacher[n]}")

    best_model_teacher = NGramModel(best_n_teacher)
    best_model_teacher.train(train_set)

    results_teacher = generate_predictions(best_model_teacher, random_test_inputs)

    perp = calculate_perplexity(best_model_teacher, test_set)
    print(f"Best-performing n test-set perplexity: {perp}")

    with open("results_teacher_model.json", "w") as f:
        json.dump(results_teacher, f, indent=4)

    print("Predictions saved successfully!")

if __name__ == "__main__":
    main()
