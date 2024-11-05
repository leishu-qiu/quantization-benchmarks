from datasets import load_dataset

# Load the Quora dataset
dataset = load_dataset("quora", split="train")

# Display the first 10 rows of the dataset
for row in dataset.select(range(10)):
    print(row)
