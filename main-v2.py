import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load the dataset (use only a subset for faster processing)
@st.cache_data
def load_data(n_rows=10000):
    df = pd.read_csv('RAW_recipes.csv', nrows=n_rows)
    if isinstance(df['tags'].iloc[0], str):
        df['tags'] = df['tags'].apply(ast.literal_eval)  # Only convert if it's a string
    return df

# Function to combine features for a specific group of tags
def combine_group_features(row, tag_column):
    if tag_column in row and isinstance(row[tag_column], list):
        return ' '.join(row[tag_column])
    return ''

# Function to calculate and return recommendations for a specific group
def get_recommendations_for_group(recipes_df, tag_column, recipe_index, top_n=10):
    # Combine features for the specified tag column
    if tag_column not in recipes_df:
        raise ValueError(f"Tag column '{tag_column}' does not exist in the DataFrame.")

    recipes_df[f'{tag_column}_features'] = recipes_df.apply(
        lambda row: combine_group_features(row, tag_column), axis=1
    )
    # Vectorize the group-specific features
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '))
    features_matrix = vectorizer.fit_transform(recipes_df[f'{tag_column}_features'])
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(features_matrix)
    # Convert similarity matrix to DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=recipes_df.index, columns=recipes_df.index)
    # Get top N similar recipes
    similar_scores = similarity_df[recipe_index].sort_values(ascending=False)
    similar_recipes = similar_scores.drop(recipe_index).head(top_n)
    return similar_recipes

# Load dataset
recipes_df = load_data()

# Define tag categories
tag_categories = {
    "time_tags": ['60-minutes-or-less', '30-minutes-or-less', '4-hours-or-less', '15-minutes-or-less', '1-day-or-more'],
    "course_tags": ['main-dish', 'side-dishes', 'desserts', 'appetizers', 'salads', 'snacks', 'soups-stews', 'breakfast', 'lunch', 'dinner-party', 'brunch'],
    "cuisine_tags": ['north-american', 'european', 'mexican', 'italian', 'asian', 'middle-eastern', 'greek', 'southern-united-states', 'indian', 'caribbean', 'african'],
    "dietary_tags": ['vegetarian', 'vegan', 'low-carb', 'low-fat', 'low-sodium', 'gluten-free', 'low-calorie', 'diabetic', 'low-cholesterol'],
    "ingredient_tags": ['vegetables', 'meat', 'seafood', 'fruit', 'eggs-dairy', 'pasta', 'poultry', 'beef', 'pork', 'nuts', 'cheese'],
    "taste_mood_tags": ['sweet', 'savory', 'spicy', 'comfort-food', 'romantic', 'healthy', 'kid-friendly', 'beginner-cook'],
    "occasion_tags": ['holiday-event', 'christmas', 'thanksgiving', 'valentines-day', 'summer', 'fall', 'winter', 'spring', 'picnic', 'to-go'],
    "technique_tags": ['oven', 'stove-top', 'grilling', 'crock-pot-slow-cooker', 'broil', 'baking', 'stir-fry'],
    "equipment_tags": ['food-processor-blender', 'small-appliance', 'refrigerator', 'microwave'],
    "health_tags": ['low-saturated-fat', 'high-calcium', 'high-protein', 'very-low-carbs', 'low-sugar'],
    "specific_ingredients_tags": ['chocolate', 'berries', 'tropical-fruit', 'onions', 'citrus', 'potatoes', 'carrots', 'mushrooms']
}

# Categorize tags into columns
for category, category_tags in tag_categories.items():
    recipes_df[category] = recipes_df['tags'].apply(lambda tags: [tag for tag in tags if tag in category_tags])

# Streamlit UI for multi-tag recommendations
st.title("Multi-Level Recipe Recommendation System")

# User selects a recipe
selected_recipe_name = st.selectbox("Select a Recipe", recipes_df['name'])

if selected_recipe_name:
    # Get the index of the selected recipe
    selected_recipe_index = recipes_df[recipes_df['name'] == selected_recipe_name].index[0]
    st.subheader(f"Selected Recipe: {selected_recipe_name}")
    st.write(f"Tags: {recipes_df.loc[selected_recipe_index, 'tags']}")

    # User selects a tag group for recommendations
    selected_group = st.selectbox("Select Tag Group for Recommendations", list(tag_categories.keys()))

    if st.button("Generate Recommendations"):
        try:
            # Generate recommendations for the selected tag group
            recommendations = get_recommendations_for_group(recipes_df, selected_group, selected_recipe_index, top_n=5)
            st.subheader(f"Recommendations Based on {selected_group.replace('_', ' ').capitalize()}:")
            for idx in recommendations.index:
                st.write(f"- **{recipes_df.loc[idx, 'name']}**")
                st.write(f"Tags: {recipes_df.loc[idx, selected_group]}")
        except ValueError as e:
            st.error(f"No valid features for {selected_group}: {e}")
