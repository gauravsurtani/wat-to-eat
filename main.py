import streamlit as st
import pandas as pd
import ast

# Sample DataFrame (Replace this with your actual recipes_df)
data = {
    'name': ['Recipe 1', 'Recipe 2', 'Recipe 3'],
    'tags': [
        "['vegetarian', 'low-carb', '30-minutes-or-less']",
        "['vegan', 'low-fat', '15-minutes-or-less']",
        "['gluten-free', 'high-protein', '60-minutes-or-less']"
    ]
}
recipes_df = pd.DataFrame(data)

# Define tag categories
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

# Streamlit UI
st.title("Recipe Filter and Recommendations")
st.sidebar.header("Filter Recipes by Tags")

# Selected tags from all categories
selected_tags = []

# Loop through tag categories and create checkboxes
for category, tags in tag_categories.items():
    with st.sidebar.expander(category, expanded=True):
        st.write(f"Select tags for **{category}**:")
        selected = st.multiselect(f"{category}", tags, key=category)
        selected_tags.extend(selected)

# Filter the DataFrame based on selected tags
if selected_tags:
    def filter_by_tags(row, tags):
        row_tags = ast.literal_eval(row['tags'])  # Convert string list to actual list
        return any(tag in row_tags for tag in tags)

    filtered_recipes = recipes_df[recipes_df.apply(filter_by_tags, tags=selected_tags, axis=1)]
else:
    filtered_recipes = recipes_df

# Display filtered recipes
st.subheader("Filtered Recipes")
if not filtered_recipes.empty:
    for _, recipe in filtered_recipes.iterrows():
        st.write(f"**{recipe['name']}** - Tags: {recipe['tags']}")
else:
    st.write("No recipes match the selected tags.")

# Function for dummy recommendations (adjust with your logic)
def get_recommendations_for_group(recipes_df, tag_column, recipe_index, top_n=5):
    return recipes_df.iloc[:top_n]  # Replace with actual recommendation logic

# Add recommendations based on selected tags
if not filtered_recipes.empty:
    st.subheader("Recommendations Based on Selected Tags")
    sample_index = filtered_recipes.index[0]
    try:
        recommendations = get_recommendations_for_group(
            filtered_recipes, 'tags', sample_index, top_n=5
        )
        st.write("Recommended Recipes:")
        for _, rec in recommendations.iterrows():
            st.write(f"- **{rec['name']}** ({rec['tags']})")
    except Exception as e:
        st.write(f"Error generating recommendations: {e}")
