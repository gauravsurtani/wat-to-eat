import pandas as pd
from recommendations import generate_group_recommendations
import random
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")


# Load the preprocessed recipes and interactions
recipes_df = pd.read_csv('preprocessed_recipes.csv')
interactions_df = pd.read_csv('filtered_interactions.csv')

# Use a smaller sample for testing
# recipes_df = recipes_df.head(5000)

# Function to get the recipe name by ID
def get_recipe_name_by_id(recipe_id, recipes_df):
    recipe = recipes_df[recipes_df['id'] == recipe_id]
    if not recipe.empty:
        return recipe['name'].iloc[0]
    return None

def get_recipe_details(recipe_id, recipes_df):
    """
    Fetch and display detailed information about a recipe ID, including its tags.

    Args:
        recipe_id (int): The recipe ID to fetch details for.
        recipes_df (pd.DataFrame): The DataFrame containing recipe information.

    Returns:
        dict: A dictionary containing the recipe's name and tags, or None if not found.
    """
    recipe = recipes_df[recipes_df['id'] == recipe_id]
    if not recipe.empty:
        details = {
            "Recipe Name": recipe['name'].iloc[0],
            "Recipe ID": recipe_id,
            "Tags": {
                "time_tags": recipe['time_tags'].iloc[0],
                "course_tags": recipe['course_tags'].iloc[0],
                "cuisine_tags": recipe['cuisine_tags'].iloc[0],
                "dietary_tags": recipe['dietary_tags'].iloc[0],
                "ingredient_tags": recipe['ingredient_tags'].iloc[0],
                "taste_mood_tags": recipe['taste_mood_tags'].iloc[0],
                "occasion_tags": recipe['occasion_tags'].iloc[0],
                "technique_tags": recipe['technique_tags'].iloc[0],
                "equipment_tags": recipe['equipment_tags'].iloc[0],
                "Health_tags": recipe['Health_tags'].iloc[0],
                "specific_ingredients_tags": recipe['specific_ingredients_tags'].iloc[0],
            }
        }
        return details
    return None



# Define tag groups for recommendations
tag_groups = [
    'time_tags', 'course_tags', 'cuisine_tags', 'dietary_tags',
    'ingredient_tags', 'taste_mood_tags', 'occasion_tags',
    'technique_tags', 'equipment_tags', 'Health_tags', 'specific_ingredients_tags'
]

available_recipe_ids = recipes_df['id'].tolist()
selected_recipe_id = random.choice(available_recipe_ids)

recipe_details = get_recipe_details(selected_recipe_id, recipes_df)

# Display details
if recipe_details:
    print(f"Recipe Name: {recipe_details['Recipe Name']}")
    print(f"Recipe ID: {recipe_details['Recipe ID']}")
    print("Tags:")
    for tag_group, tags in recipe_details["Tags"].items():
        print(f"  {tag_group}: {tags}")
else:
    print(f"Recipe ID {selected_recipe_id} not found in the dataset.")

selected_recipe_index = recipes_df[recipes_df['id'] == selected_recipe_id].index[0]

# Generate recommendations
recommendations = generate_group_recommendations(recipes_df, selected_recipe_index, tag_groups, top_n=5)

# Step 1: Identify users who liked the selected recipe
liked_users = interactions_df[
    (interactions_df['recipe_id'] == selected_recipe_id) & (interactions_df['rating'] >= 2)
]['user_id'].unique()
print(f"Users who liked Recipe ID {selected_recipe_id}: {liked_users}")

# Step 2: Find other recipes liked by these users
liked_recipes = interactions_df[
    (interactions_df['user_id'].isin(liked_users)) & (interactions_df['rating'] >= 2)
]['recipe_id'].unique()
print(f"Recipes liked by these users: {liked_recipes}")

# Step 3: Evaluate relevance of recommendations
evaluation_results = {}
for group, recs in recommendations.items():
    recommended_ids = recipes_df.loc[recs.index, 'id'].values
    matches = set(recommended_ids) & set(liked_recipes)
    evaluation_results[group] = {
        'total_recommendations': len(recommended_ids),
        'matches': len(matches),
        'match_ids': list(matches)
    }

# Display matching recipe names for each tag group
print("\nEvaluation of Recommendations with Recipe Names:")
for group, results in evaluation_results.items():
    print(f"\n{group} Recommendations:")
    print(f"  Total Recommendations: {results['total_recommendations']}")
    print(f"  Matches with liked recipes: {results['matches']}")
    if results['match_ids']:
        print("  Matching Recipes:")
        for recipe_id in results['match_ids']:
            recipe_name = get_recipe_name_by_id(recipe_id, recipes_df)
            print(f"    - Recipe ID {recipe_id}: {recipe_name}")
    else:
        print("  No matching recipes.")




# import pandas as pd
# import random
# from recommendations import generate_group_recommendations

# # Load the preprocessed recipes DataFrame
# recipes_df = pd.read_csv('preprocessed_recipes.csv')

# # Use a smaller sample for testing
# # recipes_df = recipes_df.head(5000)

# # Define tag groups for recommendations
# tag_groups = [
#     'time_tags', 'course_tags', 'cuisine_tags', 'dietary_tags',
#     'ingredient_tags', 'taste_mood_tags', 'occasion_tags',
#     'technique_tags', 'equipment_tags', 'Health_tags', 'specific_ingredients_tags'
# ]

# # Select a random recipe ID for testing
# available_recipe_ids = recipes_df['id'].tolist()
# sample_recipe_id = random.choice(available_recipe_ids)

# # Map recipe ID to the DataFrame index
# if sample_recipe_id in recipes_df['id'].values:
#     sample_recipe_index = recipes_df[recipes_df['id'] == sample_recipe_id].index[0]
# else:
#     raise ValueError(f"Recipe ID {sample_recipe_id} not found in the dataset.")

# # Print the selected recipe's details
# print(f"Selected Recipe ID: {sample_recipe_id}")
# selected_recipe = recipes_df.loc[sample_recipe_index]
# print(f"Selected Recipe Name: {selected_recipe['name']}")
# print(f"Selected Recipe Tags:")
# for tag_group in tag_groups:
#     if tag_group in selected_recipe:
#         print(f"{tag_group}: {selected_recipe[tag_group]}")

# # Generate recommendations
# recommendations = generate_group_recommendations(recipes_df, sample_recipe_index, tag_groups, top_n=5)

# # Print aggregated recommendations
# print("\nSummary of Recommendations:")
# for group, recs in recommendations.items():
#     print(f"\n{group} Recommendations:")
#     print(recipes_df.loc[recs.index, ['name', group]])





# import pandas as pd

# # Load the preprocessed recipes DataFrame
# recipes_df = pd.read_csv('preprocessed_recipes.csv')

# # Function to display all tags for a specific recipe ID
# def show_recipe_tags(recipe_id, recipes_df):
#     """
#     Display all tags associated with a specific recipe ID.

#     Args:
#         recipe_id (int): The recipe ID to search for.
#         recipes_df (pd.DataFrame): The DataFrame containing recipe data.

#     Returns:
#         None: Prints the tags and relevant columns for the recipe.
#     """
#     if recipe_id in recipes_df['id'].values:  # Check using 'id'
#         recipe_data = recipes_df.loc[recipes_df['id'] == recipe_id]  # Filter by 'id'
#         print(f"Recipe Name: {recipe_data['name'].iloc[0]}")
#         print(f"Recipe ID: {recipe_id}")
#         print("\nTags:")
#         for column in [
#             'time_tags', 'course_tags', 'cuisine_tags', 'dietary_tags',
#             'ingredient_tags', 'taste_mood_tags', 'occasion_tags',
#             'technique_tags', 'equipment_tags', 'Health_tags', 'specific_ingredients_tags'
#         ]:
#             if column in recipe_data.columns:
#                 print(f"{column}: {recipe_data[column].iloc[0]}")
#     else:
#         print(f"Recipe ID {recipe_id} not found in the dataset.")

# # Test the function with a specific recipe ID
# sample_recipe_id = 15846  # Replace with your desired recipe ID
# show_recipe_tags(sample_recipe_id, recipes_df)
