import pandas as pd

# Use the exact path you provided
file_path = "archive\BengaliEmpatheticConversationsCorpus .csv"

try:
    df = pd.read_csv(file_path)
    print("Successfully loaded the CSV file.")
    print("\nColumn names in the CSV file are:")
    print(df.columns)
except FileNotFoundError:
    print(f"Error: The file was not found at '{file_path}'. Please check the path and filename.")
except Exception as e:
    print(f"An error occurred: {e}")
