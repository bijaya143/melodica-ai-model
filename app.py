import pandas as pd
import pickle

# Function to reduce memory usage
def reduce_memory_usage(df):
    print("Reducing memory usage...")
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:  # If not a string/object column
            if col_type in ['int64', 'int']:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif col_type in ['float64', 'float']:
                df[col] = pd.to_numeric(df[col], downcast='float')
        else:
            df[col] = df[col].astype('category')  # Convert objects to categories
    print("Memory usage reduced!")
    return df

# Step 1: Load your CSV file
csv_file_path = "dataset.csv"  # Replace with your dataset's path
print("Loading dataset...")
data = pd.read_csv(csv_file_path)
print(f"Dataset loaded with shape: {data.shape}")

# Step 2: Optimize memory usage
data = reduce_memory_usage(data)

# Step 3: Save as a pickle file
pickle_file_path = "df.pkl"  # Desired pickle file name
print("Saving dataset to pickle file...")
with open(pickle_file_path, "wb") as f:
    pickle.dump(data, f)
print(f"Pickle file saved as: {pickle_file_path}")
