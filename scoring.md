F## Formal Definition:

Our evaluation metric is the rouge score between the system output and the gold standard. This is a commonly used score to find a way to assess the quality of generated summaries. 

## Relevant papers/links:

[ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf): This paper provides a justification for ROUGE-N as an evaluation metric for the text summarization task. 

## Running Evaluation Script


To run our evaluation script, you can mimic the format below, where A is your system output, B is the gold standard, and C is your n value.

```python3 scoring.py --system_output "A" --gold_standard "B" --n "C"```

For example:
```
python3 scoring.py --system_output "I enjoy vintage books and movies so I enjoyed reading this book." --gold_standard "Nice vintage story" --n "1"

ROUGE-N SCORE: 0.13333333333333333
```

