import argparse
from collections import Counter
import csv

from simple_baseline import text_rank


def get_ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def evaluation(system_output, gold_standard, n):
    # ROUGE-N is a component of the ROUGE score that quantifies the overlap of
    # N-grams, contiguous sequences of N items (typically words or characters),
    # between the system-generated summary and the reference summary.
    # It provides insights into the precision and recall of the system's output
    # by considering the matching N-gram sequences.

    # Tokenize input sentences into words
    system_tokens = system_output.split()
    gold_tokens = gold_standard.split()

    # Create n-grams from the tokenized sentences
    system_ngrams = list(get_ngrams(system_tokens, n))
    gold_ngrams = list(get_ngrams(gold_tokens, n))

    # Count occurrences of n-grams
    system_ngram_counts = Counter(system_ngrams)
    gold_ngram_counts = Counter(gold_ngrams)

    # Calculate precision, recall, and F1
    intersection = sum((system_ngram_counts & gold_ngram_counts).values())
    system_total = sum(system_ngram_counts.values())
    gold_total = sum(gold_ngram_counts.values())

    precision = intersection / system_total if system_total > 0 else 0
    recall = intersection / gold_total if gold_total > 0 else 0

    # Avoid division by zero for precision + recall = 0
    f1 = (2 * precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    return f1

# Not used in script


def evaluate_model(n):
    file_path = 'kindle_reviews.csv'

    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        score_sum = 0
        count = 0
        for line_num, row in enumerate(csv_reader):
            review_text = row['reviewText']
            gold_standard = row['summary']
            system_output = text_rank(review_text, 1)

            score_sum += evaluation(system_output, gold_standard, n)
            count += 1

            if line_num % 100 == 0:
                print(
                    f"Line {line_num} done. Current Score is: {score_sum / count}. Count={count}, score_sum={score_sum}")

    average_score = score_sum / count
    print(
        f"Final Score is: {average_score}. Count={count}, score_sum={score_sum}")

    return average_score


evaluate_model(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ROUGE evaluation script for text summarization."
    )
    parser.add_argument("--system_output", type=str,
                        help="System-generated summary")
    parser.add_argument("--gold_standard", type=str,
                        help="Gold standard summary")
    parser.add_argument("--n", type=str, help="n for n-grams to use in Rouge")

    args = parser.parse_args()

    rouge = evaluation(
        args.system_output, args.gold_standard, int(args.n))

    print(f"ROUGE-N SCORE: {rouge}")
