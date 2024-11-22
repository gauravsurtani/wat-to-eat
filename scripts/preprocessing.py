import pandas as pd
import ast

def preprocess_data(recipes_df, reviews_df, min_interactions=3):
    """
    Preprocess the recipes DataFrame with nutritional and tag information. Filters recipes
    based on minimum interactions from the reviews DataFrame.

    Args:
        recipes_df (pd.DataFrame): The recipes DataFrame containing 'nutrition' and 'tags'.
        reviews_df (pd.DataFrame): The reviews DataFrame containing user-recipe interactions.
        min_interactions (int): Minimum number of interactions required to keep a recipe.

    Returns:
        pd.DataFrame: Preprocessed and filtered recipes DataFrame.
    """

    # Step 0 : Filter Interactions Dataset
    print("Filtering interactions based on recipe interactions...")
    recipe_counts = reviews_df['recipe_id'].value_counts()
    valid_recipes = recipe_counts[recipe_counts >= min_interactions].index
    reviews_df = reviews_df[reviews_df['recipe_id'].isin(valid_recipes)]
    print(f"Filtered interactions to {len(reviews_df)} rows.")

    # Step 1: Filter recipes based on interactions
    print("Filtering recipes based on interactions...")
    recipe_counts = reviews_df['recipe_id'].value_counts()
    valid_recipes = recipe_counts[recipe_counts >= min_interactions].index
    recipes_df = recipes_df[recipes_df['id'].isin(valid_recipes)]
    print(f"Filtered recipes to {len(recipes_df)} rows based on interactions.")

    # Step 2: Process nutritional data
    print("Processing nutritional data...")
    recipes_df[['calories', 'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 
                'protein (PDV)', 'saturated fat (PDV)', 'carbohydrates (PDV)']] = (
        recipes_df['nutrition']
        .str.strip('[]')  # Remove brackets
        .str.split(",", expand=True)  # Split into individual fields
        .astype(float)  # Convert to float
    )

    # Step 3: Add custom nutritional tags
    print("Adding nutritional tags...")
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

    # Step 4: Combine health tags
    print("Combining health tags...")

    def combine_tags(row):
        categories = []  # Start with an empty list
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
        return categories  # Return the list directly

    # Apply the function to the DataFrame
    recipes_df['Health_tags'] = recipes_df.apply(combine_tags, axis=1)


    # Step 5: Categorize tags into predefined groups
    print("Categorizing tags...")
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

    def categorize_tags(tags, category_list):
        tags = ast.literal_eval(tags)  # Convert string representation of list to actual list
        return [tag for tag in tags if tag in category_list]

    for category, category_tags in tag_categories.items():
        recipes_df[category] = recipes_df['tags'].apply(lambda x: categorize_tags(x, category_tags))
    
    print("Dropping unnecessary columns...")
    columns_to_drop = [
        "minutes", "contributor_id", "submitted", "tags", "nutrition", "n_steps", 
        "steps", "description", "ingredients", "n_ingredients", "calories", 
        "total fat (PDV)", "sugar (PDV)", "sodium (PDV)", "protein (PDV)", 
        "saturated fat (PDV)", "carbohydrates (PDV)", "High Protein", 
        "Very High Protein", "Low Sugar", "Low Fat", "Low Calorie", 
        "High Calorie", "High Fat", "Moderate Sodium", "Low Sodium", 
        "Weight Loss", "Weight Gain", "Muscle Building", 
        "Blood Pressure Management"
    ]
    recipes_df = recipes_df.drop(columns=columns_to_drop, errors='ignore')
    print(f"Dropped columns: {columns_to_drop}")

    print("Preprocessing complete.")
    return recipes_df
