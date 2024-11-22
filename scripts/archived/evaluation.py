import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import ast
from collections import Counter

def preprocess_and_tag_recipes(recipes_df):
    """
    Preprocess and tag the recipes DataFrame with nutritional, course, cuisine, dietary, 
    and custom health tags.
    
    Args:
        recipes_df (pd.DataFrame): DataFrame containing recipe information including 'nutrition' and 'tags'.
    
    Returns:
        pd.DataFrame: The modified DataFrame with new tag columns.
    """
    # Preprocess nutritional data
    recipes_df[['calories', 'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 
                'protein (PDV)', 'saturated fat (PDV)', 'carbohydrates (PDV)']] = recipes_df['nutrition'].str.split(",", expand=True)
    recipes_df['calories'] = recipes_df['calories'].apply(lambda x: x.replace('[', '') if isinstance(x, str) else x)
    recipes_df['carbohydrates (PDV)'] = recipes_df['carbohydrates (PDV)'].apply(lambda x: x.replace(']', '') if isinstance(x, str) else x)
    recipes_df[['calories', 'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 
                'protein (PDV)', 'saturated fat (PDV)', 'carbohydrates (PDV)']] = recipes_df[['calories', 'total fat (PDV)', 
                                                                                            'sugar (PDV)', 'sodium (PDV)', 
                                                                                            'protein (PDV)', 'saturated fat (PDV)', 
                                                                                            'carbohydrates (PDV)']].astype('float')

    # Add custom nutritional tags
    recipes_df['High Protein'] = (recipes_df['protein (PDV)'] > 20).astype(int)
    recipes_df['Very High Protein'] = (recipes_df['protein (PDV)'] > 30).astype(int)
    recipes_df['Low Sugar'] = (recipes_df['sugar (PDV)'] < 5).astype(int)
    recipes_df['Low Fat'] = (recipes_df['total fat (PDV)'] < 10).astype(int)
    recipes_df['Low Calorie'] = (recipes_df['calories'] < 200).astype(int)
    recipes_df['High Calorie'] = (recipes_df['calories'] > 300).astype(int)
    recipes_df['High Fat'] = (recipes_df['total fat (PDV)'] > 15).astype(int)
    recipes_df['Moderate Sodium'] = ((recipes_df['sodium (PDV)'] >= 10) & (recipes_df['sodium (PDV)'] <= 15)).astype(int)
    recipes_df['Low Sodium'] = (recipes_df['sodium (PDV)'] < 5).astype(int)
    recipes_df['Weight Loss'] = (recipes_df['Low Sugar'] + recipes_df['Low Calorie'] + recipes_df['Low Fat'] >= 2).astype(int)
    recipes_df['Weight Gain'] = (recipes_df['High Calorie'] + recipes_df['High Protein'] >= 2).astype(int)
    recipes_df['Muscle Building'] = (recipes_df['High Protein'] + recipes_df['High Fat'] >= 2).astype(int)
    recipes_df['Blood Pressure Management'] = (recipes_df['Low Sodium'] + recipes_df['Moderate Sodium'] >= 1).astype(int)

    # Combine health tags
    def combine_tags(row):
        categories = []
        if row['Weight Loss']:
            categories.append('Weight Loss')
        if row['Weight Gain']:
            categories.append('Weight Gain')
        if row['Muscle Building']:
            categories.append('Muscle Building')
        if row['Blood Pressure Management']:
            categories.append('Blood Pressure Management')
        if row['High Protein']:
            categories.append('High Protein')
        if row['Very High Protein']:
            categories.append('Very High Protein')
        if row['Low Sugar']:
            categories.append('Low Sugar')
        if row['Low Fat']:
            categories.append('Low Fat')
        if row['Low Calorie']:
            categories.append('Low Calorie')
        if row['High Calorie']:
            categories.append('High Calorie')
        if row['High Fat']:
            categories.append('High Fat')
        if row['Moderate Sodium']:
            categories.append('Moderate Sodium')
        if row['Low Sodium']:
            categories.append('Low Sodium')
        if not categories:
            categories.append('General')  # Default category
        return ', '.join(categories)
    
    recipes_df['Health_tags'] = recipes_df.apply(combine_tags, axis=1)

    # Define tag categories
    tag_categories = {
        "time_tags": ['60-minutes-or-less', '30-minutes-or-less', '4-hours-or-less', '15-minutes-or-less', '1-day-or-more'],
        "course_tags": ['main-dish', 'side-dishes', 'desserts', 'appetizers', 'salads', 'snacks', 'soups-stews', 
                        'breakfast', 'lunch', 'dinner-party', 'brunch'],
        "cuisine_tags": ['north-american', 'european', 'mexican', 'italian', 'asian', 'middle-eastern', 'greek', 
                         'southern-united-states', 'indian', 'caribbean', 'african'],
        "dietary_tags": ['vegetarian', 'vegan', 'low-carb', 'low-fat', 'low-sodium', 'gluten-free', 'low-calorie', 
                         'diabetic', 'low-cholesterol'],
        "ingredient_tags": ['vegetables', 'meat', 'seafood', 'fruit', 'eggs-dairy', 'pasta', 'poultry', 'beef', 
                            'pork', 'nuts', 'cheese'],
        "taste_mood_tags": ['sweet', 'savory', 'spicy', 'comfort-food', 'romantic', 'healthy', 'kid-friendly', 
                            'beginner-cook'],
        "occasion_tags": ['holiday-event', 'christmas', 'thanksgiving', 'valentines-day', 'summer', 'fall', 
                          'winter', 'spring', 'picnic', 'to-go'],
        "technique_tags": ['oven', 'stove-top', 'grilling', 'crock-pot-slow-cooker', 'broil', 'baking', 'stir-fry'],
        "equipment_tags": ['food-processor-blender', 'small-appliance', 'refrigerator', 'microwave'],
        "specific_ingredients_tags": ['chocolate', 'berries', 'tropical-fruit', 'onions', 'citrus', 
                                      'potatoes', 'carrots', 'mushrooms']
    }

    # Categorize tags into columns
    for category, category_tags in tag_categories.items():
        recipes_df[category] = recipes_df['tags'].apply(lambda tags: categorize_tags(tags, category_tags))

    return recipes_df

# Helper function to categorize tags into predefined categories
def categorize_tags(tags, category_list):
    if isinstance(tags, str):  # Ensure tags are a list
        tags = ast.literal_eval(tags)
    return [tag for tag in tags if tag in category_list]


# Function to preprocess and filter data
def preprocess_and_filter_data(recipes_file, reviews_file, filtered_recipes_file, filtered_reviews_file, min_recipe_interactions=3, min_user_interactions=3):
    """
    Preprocess and filter data with checks for existing filtered files. If tags are missing in the
    filtered recipes file, it will process and add them.
    """
    # Check if filtered files exist
    if os.path.exists(filtered_recipes_file) and os.path.exists(filtered_reviews_file):
        print("Filtered files exist. Loading filtered data...")
        filtered_recipes_df = pd.read_csv(filtered_recipes_file)
        filtered_reviews_df = pd.read_csv(filtered_reviews_file)
        
        # Check if required tags are present
        required_tag_columns = [
            "time_tags", "course_tags", "cuisine_tags", "dietary_tags", 
            "ingredient_tags", "taste_mood_tags", "occasion_tags", 
            "technique_tags", "equipment_tags", "specific_ingredients_tags", "Health_tags"
        ]
        missing_tags = [col for col in required_tag_columns if col not in filtered_recipes_df.columns]

        if missing_tags:
            print(f"Missing tags in filtered recipes file: {missing_tags}. Adding tags...")
            # Preprocess and add tags
            filtered_recipes_df = preprocess_and_tag_recipes(filtered_recipes_df)
            # Save the updated filtered recipes file
            filtered_recipes_df.to_csv(filtered_recipes_file, index=False)
            print("Updated filtered recipes file with tags.")
        
        return filtered_recipes_df, filtered_reviews_df

    print("Filtered files not found. Preprocessing and filtering data...")
    # Load data
    recipes_df = pd.read_csv(recipes_file)
    reviews_df = pd.read_csv(reviews_file)

    # Filter recipes with a minimum number of interactions
    recipe_review_counts = reviews_df['recipe_id'].value_counts()
    valid_recipes = recipe_review_counts[recipe_review_counts >= min_recipe_interactions].index
    filtered_recipes_df = recipes_df[recipes_df['id'].isin(valid_recipes)]

    print(f"Original Recipes Shape: {recipes_df.shape}")
    print(f"Filtered Recipes Shape: {filtered_recipes_df.shape}")

    # Filter users with a minimum number of interactions
    user_interaction_counts = reviews_df['user_id'].value_counts()
    valid_users = user_interaction_counts[user_interaction_counts > min_user_interactions].index
    filtered_reviews_df = reviews_df[reviews_df['user_id'].isin(valid_users)]

    print(f"Original Reviews Shape: {reviews_df.shape}")
    print(f"Filtered Reviews Shape: {filtered_reviews_df.shape}")

    # Preprocess and tag the filtered recipes
    print("Adding tags to filtered recipes...")
    filtered_recipes_df = preprocess_and_tag_recipes(filtered_recipes_df)

    # Save filtered data for future use
    filtered_recipes_df.to_csv(filtered_recipes_file, index=False)
    filtered_reviews_df.to_csv(filtered_reviews_file, index=False)
    print("Filtered data saved with tags.")

    return filtered_recipes_df, filtered_reviews_df


# Function to find users who liked a recipe
def get_users_who_liked_recipe(reviews_df, recipe_id, rating_threshold=4):
    liked_users = reviews_df[(reviews_df['recipe_id'] == recipe_id) & (reviews_df['rating'] >= rating_threshold)]
    return liked_users['user_id'].unique()

# Function to find recipes liked by these users
def get_recipes_liked_by_users(reviews_df, user_ids, rating_threshold=4):
    liked_recipes = reviews_df[(reviews_df['user_id'].isin(user_ids)) & (reviews_df['rating'] >= rating_threshold)]
    return liked_recipes['recipe_id'].unique()

# Function to evaluate recommendations
def evaluate_recommendations(recommended_recipes, liked_recipes):
    matches = set(recommended_recipes) & set(liked_recipes)
    return matches, len(matches)

# Function to combine features for a specific group of tags
def combine_group_features(row, tag_column):
    if tag_column in row and isinstance(row[tag_column], list):
        return ' '.join(row[tag_column])
    return ''

# Function to calculate and return recommendations for a specific group using Nearest Neighbors
def get_recommendations_for_group(recipes_df, tag_column, recipe_index, top_n=10):
    """
    Generate recommendations for a specific tag group using Nearest Neighbors.

    Args:
        recipes_df (pd.DataFrame): The recipes DataFrame.
        tag_column (str): The tag column to use for recommendations.
        recipe_index (int): The index of the selected recipe.
        top_n (int): The number of recommendations to return.

    Returns:
        pd.DataFrame: DataFrame of similar recipes with similarity scores.
    """
    # Combine features for the specified tag column
    recipes_df[f'{tag_column}_features'] = recipes_df.apply(
        lambda row: combine_group_features(row, tag_column), axis=1
    )
    
    # Vectorize the group-specific features
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '))
    features_matrix = vectorizer.fit_transform(recipes_df[f'{tag_column}_features'])
    
    # Use NearestNeighbors for recommendations
    model = NearestNeighbors(n_neighbors=top_n + 1, metric='cosine', algorithm='brute')
    model.fit(features_matrix)

    # Find the nearest neighbors for the selected recipe
    distances, indices = model.kneighbors(features_matrix[recipe_index])

    # Exclude the input recipe itself (distance = 0)
    similar_indices = indices[0][1:]
    similar_scores = 1 - distances[0][1:]  # Convert distances to similarity scores

    # Return a DataFrame of similar recipes
    return pd.DataFrame({
        'index': similar_indices,
        'similarity_score': similar_scores
    }).set_index('index')


# Main evaluation workflow
def evaluate_recommendation_system(
    recipes_file, reviews_file, filtered_recipes_file, filtered_reviews_file, 
    selected_recipe_id, min_recipe_interactions=3, min_user_interactions=3
):
    # Preprocess and filter data
    recipes_df, reviews_df = preprocess_and_filter_data(
        recipes_file, reviews_file, 
        filtered_recipes_file, filtered_reviews_file,
        min_recipe_interactions=min_recipe_interactions, 
        min_user_interactions=min_user_interactions
    )

    # Convert tags to lists
    recipes_df['tags'] = recipes_df['tags'].apply(ast.literal_eval)

    # Step 1: Get users who liked the selected recipe
    liked_users = get_users_who_liked_recipe(reviews_df, selected_recipe_id, rating_threshold=4)

    # Step 2: Get other recipes liked by these users
    other_liked_recipes = get_recipes_liked_by_users(reviews_df, liked_users, rating_threshold=4)

    # Step 3: Generate multi-level recommendations
    tag_groups = [
        'time_tags', 'course_tags', 'cuisine_tags', 'dietary_tags',
        'ingredient_tags', 'taste_mood_tags', 'occasion_tags',
        'technique_tags', 'equipment_tags', 'health_tags', 'specific_ingredients_tags'
    ]

    sample_recipe_index = recipes_df[recipes_df['id'] == selected_recipe_id].index[0]
    # Initialize a Counter to track recommendation frequencies
    recommendation_counter = Counter()

    print("\nGenerating Recommendations for Each Tag Group:")

    # Aggregate recommendations across all tag groups
    for group in tag_groups:
        print(f"\nProcessing recommendations for tag group: {group}")
        try:
            recommendations = get_recommendations_for_group(
                recipes_df, group, sample_recipe_index, top_n=5
            )

            # Print recommendations for this group
            print(f"Top recommendations for {group}:")
            for recipe_id in recommendations.index:
                matching_recipes = recipes_df[recipes_df['id'] == recipe_id]
                if not matching_recipes.empty:
                    recipe_name = matching_recipes['name'].iloc[0]
                    print(f"  - Recipe ID {recipe_id}: {recipe_name}")
                else:
                    print(f"  - Recipe ID {recipe_id}: Name not found")


            # Add recommendations to the counter
            recommendation_counter.update(recommendations.index)
        except ValueError:
            print(f"Skipping {group} due to an error.")
            continue

    # Sort recipes by recommendation count (most recommended first)
    most_recommended_recipes = recommendation_counter.most_common(10)

    print("\nAggregated Recommendations Across All Tag Groups:")
    for recipe_id, count in most_recommended_recipes:
        matching_recipes = recipes_df[recipes_df['id'] == recipe_id]
        if not matching_recipes.empty:
            recipe_name = matching_recipes['name'].iloc[0]
            print(f"- Recipe ID {recipe_id}: {recipe_name} (Recommended {count} times)")
        else:
            print(f"- Recipe ID {recipe_id}: Name not found (Recommended {count} times)")

    # Step 4: Evaluate recommendations
    matches, num_matches = evaluate_recommendations(
        [recipe_id for recipe_id, _ in most_recommended_recipes], 
        other_liked_recipes
    )

    print("\nEvaluation of Recommendations:")
    # Display results
    print(f"Users who liked Recipe ID {selected_recipe_id}: {liked_users}")
    print(f"Recipes liked by these users: {other_liked_recipes}")
    print(f"Recommended Recipes: {[recipe_id for recipe_id, _ in most_recommended_recipes]}")
    print(f"Number of Matches: {num_matches}")
    print(f"Matching Recipes: {matches}")

    # Matching Recipe Names
    print("\nMatching Recipe Names:")
    matching_recipe_names = [
        recipes_df[recipes_df['id'] == recipe_id]['name'].iloc[0] 
        for recipe_id in matches 
        if not recipes_df[recipes_df['id'] == recipe_id].empty
    ]
    for name in matching_recipe_names:
        print(f"- {name}")


# Entry point
if __name__ == "__main__":
    # File paths
    recipes_file = 'RAW_recipes.csv'
    reviews_file = 'RAW_interactions.csv'
    filtered_recipes_file = 'filtered_recipes.csv'
    filtered_reviews_file = 'filtered_reviews.csv'

    # Selected recipe ID for testing
    selected_recipe_id = 15846  # Replace with your recipe ID

    # Run the evaluation
    evaluate_recommendation_system(
        recipes_file, 
        reviews_file, 
        filtered_recipes_file, 
        filtered_reviews_file, 
        selected_recipe_id
    )
