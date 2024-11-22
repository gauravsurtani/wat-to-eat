import pandas as pd
from preprocessing import preprocess_data

# Load the recipe and interaction data
recipes_df = pd.read_csv('RAW_recipes.csv')
interactions_df = pd.read_csv('RAW_interactions.csv')

# Filter interactions to only keep recipes with at least 3 interactions
recipe_counts = interactions_df['recipe_id'].value_counts()
valid_recipes = recipe_counts[recipe_counts >= 3].index
filtered_interactions_df = interactions_df[interactions_df['recipe_id'].isin(valid_recipes)]

# Further filter recipes to include only those in the interactions
filtered_recipes_df = recipes_df[recipes_df['id'].isin(filtered_interactions_df['recipe_id'])]

# Preprocess the filtered recipes
processed_recipes_df = preprocess_data(filtered_recipes_df, filtered_interactions_df)

# Save the preprocessed DataFrame and filtered interactions DataFrame to CSV files
processed_recipes_df.to_csv('preprocessed_recipes.csv', index=False)
filtered_interactions_df.to_csv('filtered_interactions.csv', index=False)

print("Preprocessed recipes saved to 'preprocessed_recipes.csv'.")
print("Filtered interactions saved to 'filtered_interactions.csv'.")
