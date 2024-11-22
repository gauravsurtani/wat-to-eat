from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast

# Function to combine features for a specific group of tags
def combine_group_features(row, tag_column):
    """
    Combine features for a specific tag group into a single string.
    """
    if tag_column in row and isinstance(row[tag_column], (list, str)):
        tags = row[tag_column]
        if isinstance(tags, str):
            tags = ast.literal_eval(tags)  # Convert string representation to list
        return ' '.join(tags) if tags else 'no_tags'
    return 'no_tags'

# Function to calculate and return recommendations for a specific group
def get_recommendations_for_group(recipes_df, tag_column, recipe_index, top_n=10):
    """
    Generate recommendations for a specific tag group using cosine similarity.
    """
    # Combine features for the specified tag column
    recipes_df[f'{tag_column}_features'] = recipes_df.apply(
        lambda row: combine_group_features(row, tag_column), axis=1
    )

    # Check if there are valid features for the column
    if recipes_df[f'{tag_column}_features'].str.strip().eq('').all():
        raise ValueError(f"No valid features found for tag column: {tag_column}")

    # Vectorize the group-specific features
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '))
    features_matrix = vectorizer.fit_transform(recipes_df[f'{tag_column}_features'])

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(features_matrix)

    # Convert similarity matrix to DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=recipes_df.index, columns=recipes_df.index)

    # Get top N similar recipes excluding the input recipe itself
    similar_scores = similarity_df.iloc[recipe_index].sort_values(ascending=False)
    similar_recipes = similar_scores.drop(recipe_index).head(top_n)

    # Return a DataFrame of recommendations
    return pd.DataFrame({
        'index': similar_recipes.index,
        'similarity_score': similar_recipes.values
    }).set_index('index')

# Function to generate group-wise recommendations
def generate_group_recommendations(recipes_df, recipe_index, tag_groups, top_n=10):
    """
    Generate recommendations for each tag group and aggregate results.
    """
    all_recommendations = {}

    for group in tag_groups:
        try:
            recommendations = get_recommendations_for_group(
                recipes_df, group, recipe_index, top_n=top_n
            )
            all_recommendations[group] = recommendations
            print(f"\nRecommendations based on {group}:")
            print(recipes_df.loc[recommendations.index, ['name', group]])
        except ValueError as e:
            print(f"\nNo valid features for {group}: {e}")

    return all_recommendations
