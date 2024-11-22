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

# Function to evaluate recommendations for multiple recipes
def evaluate_recommendations_for_multiple(
    recipes_df, interactions_df, tag_groups, recipe_ids, top_n=5
):
    """
    Evaluate recommendations for multiple recipes and calculate average metrics.

    Args:
        recipes_df (pd.DataFrame): Preprocessed recipes DataFrame.
        interactions_df (pd.DataFrame): User-recipe interactions DataFrame.
        tag_groups (list): List of tag columns to generate recommendations for.
        recipe_ids (list): List of recipe IDs to evaluate.
        top_n (int): Number of top recommendations to return per group.

    Returns:
        dict: Dictionary containing overall average precision and best match score.
        pd.DataFrame: DataFrame containing evaluation details for each recipe.
    """
    overall_precision = 0
    overall_best_match_score = 0
    total_evaluations = 0
    evaluation_records = []  # To store per-recipe evaluation metrics

    print(f"Evaluating {len(recipe_ids)} recipes...")

    for recipe_id in recipe_ids:
        try:
            # Get the recipe index
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
            for group, recs in recommendations.items():
                recommended_ids = recipes_df.loc[recs.index, 'id'].values
                similarity_scores = recs['similarity_score'].values  # Extract similarity scores

                evaluation = evaluate_recommendations_with_best_score(
                    recommended_ids, liked_recipes, similarity_scores
                )

                # Update overall metrics
                overall_precision += evaluation['precision']
                overall_best_match_score += evaluation['best_match_score']
                total_evaluations += 1

                # Log individual record
                evaluation_records.append({
                    "Recipe ID": recipe_id,
                    "Recipe Name": get_recipe_name_by_id(recipe_id, recipes_df),
                    "Tag Group": group,
                    "Total Recommendations": len(recommended_ids),
                    "Matches": len(evaluation['matches']),
                    "Precision": evaluation['precision'],
                    "Best Match Score": evaluation['best_match_score']
                })

        except Exception as e:
            print(f"Error processing Recipe ID {recipe_id}: {e}")

    # Calculate averages
    average_precision = overall_precision / total_evaluations if total_evaluations > 0 else 0
    average_best_match_score = overall_best_match_score / total_evaluations if total_evaluations > 0 else 0

    print(f"\nEvaluation Complete.")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Average Best Match Score: {average_best_match_score:.4f}")

    # Convert records to DataFrame
    evaluation_df = pd.DataFrame(evaluation_records)
    return {
        "average_precision": average_precision,
        "average_best_match_score": average_best_match_score,
        "evaluation_df": evaluation_df
    }

# Main execution
if __name__ == "__main__":
    recipes_df = pd.read_csv('preprocessed_recipes.csv')
    interactions_df = pd.read_csv('filtered_interactions.csv')

    # Define tag groups
    tag_groups = [
        'time_tags', 'course_tags', 'cuisine_tags', 'dietary_tags',
        'ingredient_tags', 'taste_mood_tags', 'occasion_tags',
        'technique_tags', 'equipment_tags', 'Health_tags', 'specific_ingredients_tags'
    ]

    # Choose all recipes or a random sample
    all_recipe_ids = recipes_df['id'].tolist()
    random_sample_recipe_ids = random.sample(all_recipe_ids, k=20)  # Random sample of 100 recipes

    # Evaluate recommendations for the random sample
    results = evaluate_recommendations_for_multiple(
        recipes_df, interactions_df, tag_groups, random_sample_recipe_ids, top_n=25
    )

    # Save detailed results to CSV
    results["evaluation_df"].to_csv("recommendation_evaluation_results.csv", index=False)

    # Display overall results
    print("\nOverall Evaluation Results:")
    print(f"Average Precision: {results['average_precision']:.4f}")
    print(f"Average Best Match Score: {results['average_best_match_score']:.4f}")
