import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time


# Load the dataset (use only 10,000 rows for faster processing)
@st.cache_data
def load_data():
    df = pd.read_csv('RAW_recipes.csv')  # Load only 10,000 rows
    return df

recipes_df = load_data()

# Define tag categories
tag_columns_to_use = [
    'time_tags', 'course_tags', 'cuisine_tags', 'dietary_tags',
    'ingredient_tags', 'taste_mood_tags', 'occasion_tags',
    'technique_tags', 'equipment_tags', 'health_tags', 'specific_ingredients_tags'
]

# Function to categorize tags into columns
def categorize_tags(tags, category_list):
    tags = ast.literal_eval(tags)  # Convert string to list
    return [tag for tag in tags if tag in category_list]


tag_categories = {
    "Time Tags": ['60-minutes-or-less', '30-minutes-or-less', '4-hours-or-less', '15-minutes-or-less', '1-day-or-more'],
    "Course Tags": ['main-dish', 'side-dishes', 'desserts', 'appetizers', 'salads', 'snacks', 'soups-stews', 'breakfast', 'lunch', 'dinner-party', 'brunch'],
    "Cuisine Tags": ['north-american', 'european', 'mexican', 'italian', 'asian', 'middle-eastern', 'greek', 'southern-united-states', 'indian', 'caribbean', 'african'],
    "Dietary Tags": ['vegetarian', 'vegan', 'low-carb', 'low-fat', 'low-sodium', 'gluten-free', 'low-calorie', 'diabetic', 'low-cholesterol'],
    "Ingredient Tags": ['vegetables', 'meat', 'seafood', 'fruit', 'eggs-dairy', 'pasta', 'poultry', 'beef', 'pork', 'nuts', 'cheese'],
    "Taste & Mood Tags": ['sweet', 'savory', 'spicy', 'comfort-food', 'romantic', 'healthy', 'kid-friendly', 'beginner-cook'],
    "Occasion Tags": ['holiday-event', 'christmas', 'thanksgiving', 'valentines-day', 'summer', 'fall', 'winter', 'spring', 'picnic', 'to-go'],
    "Technique Tags": ['oven', 'stove-top', 'grilling', 'crock-pot-slow-cooker', 'broil', 'baking', 'stir-fry'],
    "Equipment Tags": ['food-processor-blender', 'small-appliance', 'refrigerator', 'microwave'],
    "Health Tags": ['low-saturated-fat', 'high-calcium', 'high-protein', 'very-low-carbs', 'low-sugar'],
    "Specific Ingredients Tags": ['chocolate', 'berries', 'tropical-fruit', 'onions', 'citrus', 'potatoes', 'carrots', 'mushrooms']
}

# Add categorized columns to the DataFrame (replace with your actual tag definitions)
recipes_df['time_tags'] = recipes_df['tags'].apply(lambda x: categorize_tags(x, tag_categories["Time Tags"]))
recipes_df['course_tags'] = recipes_df['tags'].apply(lambda x: categorize_tags(x, tag_categories["Course Tags"]))
recipes_df['cuisine_tags'] = recipes_df['tags'].apply(lambda x: categorize_tags(x, tag_categories["Cuisine Tags"]))
recipes_df['dietary_tags'] = recipes_df['tags'].apply(lambda x: categorize_tags(x, tag_categories["Dietary Tags"]))
recipes_df['ingredient_tags'] = recipes_df['tags'].apply(lambda x: categorize_tags(x, tag_categories["Ingredient Tags"]))
recipes_df['taste_mood_tags'] = recipes_df['tags'].apply(lambda x: categorize_tags(x, tag_categories["Taste & Mood Tags"]))
recipes_df['occasion_tags'] = recipes_df['tags'].apply(lambda x: categorize_tags(x, tag_categories["Occasion Tags"]))
recipes_df['technique_tags'] = recipes_df['tags'].apply(lambda x: categorize_tags(x, tag_categories["Technique Tags"]))
recipes_df['equipment_tags'] = recipes_df['tags'].apply(lambda x: categorize_tags(x, tag_categories["Equipment Tags"]))
recipes_df['health_tags'] = recipes_df['tags'].apply(lambda x: categorize_tags(x, tag_categories["Health Tags"]))
recipes_df['specific_ingredients_tags'] = recipes_df['tags'].apply(lambda x: categorize_tags(x, tag_categories["Specific Ingredients Tags"]))

# Combine selected tags into a single feature column
def combine_selected_features(row, tag_columns):
    combined_str = ''
    for col in tag_columns:
        if col in row:
            combined_str += ' '.join(row[col]) + ' '
    return combined_str.strip()

recipes_df['selected_features'] = recipes_df.apply(
    lambda row: combine_selected_features(row, tag_columns_to_use), axis=1
)

# Compute similarity matrix
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '))
selected_features_matrix = vectorizer.fit_transform(recipes_df['selected_features'])
similarity_matrix = cosine_similarity(selected_features_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=recipes_df.index, columns=recipes_df.index)

# Function to get similar recipes
def get_similar_recipes(recipe_index, top_n=5):
    similar_scores = similarity_df[recipe_index].sort_values(ascending=False)
    similar_recipes = similar_scores.drop(recipe_index).head(top_n)
    return similar_recipes

# Streamlit UI
st.title("Advanced Recipe Recommender")

# Dropdown for recipe selection
selected_recipe_name = st.selectbox("Select a Recipe", recipes_df['name'])

# Display selected recipe details
if selected_recipe_name:
    selected_recipe_index = recipes_df[recipes_df['name'] == selected_recipe_name].index[0]
    st.write(f"**Selected Recipe:** {selected_recipe_name}")
    st.write(f"**Tags:** {recipes_df.loc[selected_recipe_index, 'tags']}")

    # Fetch recommendations
    recommended_recipes = get_similar_recipes(selected_recipe_index, top_n=5)

    st.subheader("Recommended Recipes:")
    for index in recommended_recipes.index:
        st.write(f"**{recipes_df.loc[index, 'name']}**")
        st.write(f"Tags: {recipes_df.loc[index, 'tags']}")
