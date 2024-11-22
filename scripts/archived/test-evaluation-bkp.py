import pandas as pd
from recommendations import generate_group_recommendations
from evaluation import evaluate_recommendations_with_best_score
import random
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")

# Function to get the recipe name by ID
def get_recipe_name_by_id(recipe_id, recipes_df):
    """Retrieve the recipe name by its ID."""
    recipe_row = recipes_df[recipes_df['id'] == recipe_id]
    if not recipe_row.empty:
        return recipe_row.iloc[0]['name']
    return "Unknown Recipe"

# Function to fetch detailed tags for a recipe
def get_recipe_details(recipe_id, recipes_df):
    """
    Fetch detailed information about a recipe ID, including its tags.
    """
    recipe_row = recipes_df[recipes_df['id'] == recipe_id]
    if recipe_row.empty:
        return None
    return {
        "Recipe Name": recipe_row['name'].iloc[0],
        "Recipe ID": recipe_id,
        "Tags": {
            tag: recipe_row[tag].iloc[0] for tag in [
                'time_tags', 'course_tags', 'cuisine_tags', 'dietary_tags',
                'ingredient_tags', 'taste_mood_tags', 'occasion_tags',
                'technique_tags', 'equipment_tags', 'Health_tags', 'specific_ingredients_tags'
            ]
        }
    }

# Function to evaluate and print recommendations
def evaluate_and_display_recommendations(recipes_df, interactions_df, recipe_id, tag_groups, top_n=5):
    """
    Evaluate recommendations for a specific recipe and display metrics.
    """
    # Check if the recipe exists
    matched_indices = recipes_df[recipes_df['id'] == recipe_id].index
    if len(matched_indices) != 1:
        raise ValueError(f"Recipe ID {recipe_id} has ambiguous or no matches in the dataset.")
    recipe_index = matched_indices[0]

    # Generate recommendations
    recommendations = generate_group_recommendations(recipes_df, recipe_index, tag_groups, top_n=top_n)
    
    # Identify liked users and recipes
    liked_users = interactions_df[interactions_df['recipe_id'] == recipe_id]['user_id'].unique()
    liked_recipes = interactions_df[interactions_df['user_id'].isin(liked_users)]['recipe_id'].unique()

    # Evaluate relevance
    evaluation_results = {}
    for group, recs in recommendations.items():
        recommended_ids = recipes_df.loc[recs.index, 'id'].values
        similarity_scores = recs['similarity_score'].values  # Extract similarity scores

        evaluation = evaluate_recommendations_with_best_score(
            recommended_ids, liked_recipes, similarity_scores
        )
        evaluation_results[group] = evaluation

    # Display matching recipe names for each tag group
    print("\nEvaluation of Recommendations with Best Match Scores:")
    for group, results in evaluation_results.items():
        print(f"\n{group} Recommendations:")
        print(f"  Total Recommendations: {len(recommended_ids)}")
        print(f"  Matches with liked recipes: {results['num_matches']}")
        print(f"  Precision: {results['precision']:.2f}")
        # print(f"  Recall: {results['recall']:.2f}")
        print(f"  Best Match Score: {results['best_match_score']:.2f}")
        if results['matches']:
            print("  Matching Recipes:")
            for recipe_id in results['matches']:
                recipe_name = get_recipe_name_by_id(recipe_id, recipes_df)
                print(f"    - Recipe ID {recipe_id}: {recipe_name}")
        else:
            print("  No matching recipes.")

# Load preprocessed recipes and interactions
if __name__ == "__main__":
    recipes_df = pd.read_csv('preprocessed_recipes.csv')
    interactions_df = pd.read_csv('filtered_interactions.csv')

    # Define tag groups
    tag_groups = [
        'time_tags', 'course_tags', 'cuisine_tags', 'dietary_tags',
        'ingredient_tags', 'taste_mood_tags', 'occasion_tags',
        'technique_tags', 'equipment_tags', 'Health_tags', 'specific_ingredients_tags'
    ]

    # Select a random recipe ID for testing
    sample_recipe_id = random.choice(recipes_df['id'].tolist())

    # Evaluate and display recommendations for the selected recipe
    evaluate_and_display_recommendations(recipes_df, interactions_df, sample_recipe_id, tag_groups, top_n=5)
