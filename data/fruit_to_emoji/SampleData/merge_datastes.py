import os
import pandas as pd


def combine_fruit_data(directory):
    combined_data = []

    # Iterate through CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            fruit_name = os.path.splitext(filename)[0]  # Get fruit name from file name

            # Read CSV file
            df = pd.read_csv(file_path)

            # Add fruit column
            df["Fruit"] = fruit_name

            # Append to combined data
            combined_data.append(df)

    # Concatenate all dataframes
    result = pd.concat(combined_data, ignore_index=True)

    # Reorder columns to put 'Fruit' first
    columns = ["Fruit"] + [col for col in result.columns if col != "Fruit"]
    result = result[columns]

    return result


# Directory containing the CSV files
directory = "data/fruit_to_emoji/SampleData"

# Combine the data
combined_fruit_data = combine_fruit_data(directory)

# Save the combined data to a new CSV file
output_file = "fruit_data.csv"
combined_fruit_data.to_csv(f"{directory}/{output_file}", index=False)

print(f"Combined data saved to {output_file}")
