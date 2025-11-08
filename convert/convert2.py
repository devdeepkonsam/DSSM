import pandas as pd # type: ignore

# Load CSV file
df = pd.read_csv('Training_Dataset.csv')

# Function to clean byte string literals like "b'-1'" to "-1"
def clean_byte_string(val):
    if isinstance(val, str) and val.startswith("b'") and val.endswith("'"):
        return val[2:-1]  # strip b' and ending '
    return val

# Apply cleaning function to all object type columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(clean_byte_string)

# Optionally convert columns to numeric if appropriate
for col in df.columns:
    if col != 'Result':
        df[col] = pd.to_numeric(df[col], errors='ignore')

# Save cleaned data to new CSV
df.to_csv('cleanedfile.csv', index=False)

print("Cleaned CSV saved as 'cleanedfile.csv'")
