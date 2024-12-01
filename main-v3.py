import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# Load the dataset (use only a subset for faster processing)
@st.cache_data
def load_data(n_rows=10000):
    df = pd.read_csv('RAW_recipes.csv', nrows=n_rows)
    
    # Convert tags to lists if stored as strings
    if isinstance(df['tags'].iloc[0], str):
        df['tags'] = df['tags'].apply(ast.literal_eval)
    
    # Nutrition Preprocessing
    df[['calories', 'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)', 'saturated fat (PDV)', 'carbohydrates (PDV)']] = df['nutrition'].str.split(",", expand=True)
    df['calories'] = df['calories'].apply(lambda x: x.replace('[', ''))
    df['carbohydrates (PDV)'] = df['carbohydrates (PDV)'].apply(lambda x: x.replace(']', ''))
    df[['calories', 'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)', 'saturated fat (PDV)', 'carbohydrates (PDV)']] = df[[
        'calories', 'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)', 'saturated fat (PDV)', 'carbohydrates (PDV)'
    ]].astype('float')

    # Add derived tags for health categories
    df['High Protein'] = (df['protein (PDV)'] > 20).astype(int)
    df['Very High Protein'] = (df['protein (PDV)'] > 30).astype(int)
    df['Low Sugar'] = (df['sugar (PDV)'] < 5).astype(int)
    df['Low Fat'] = (df['total fat (PDV)'] < 10).astype(int)
    df['Low Calorie'] = (df['calories'] < 200).astype(int)
    df['High Calorie'] = (df['calories'] > 300).astype(int)
    df['High Fat'] = (df['total fat (PDV)'] > 15).astype(int)
    df['Moderate Sodium'] = ((df['sodium (PDV)'] >= 10) & (df['sodium (PDV)'] <= 15)).astype(int)
    df['Low Sodium'] = (df['sodium (PDV)'] < 5).astype(int)
    df['Weight Loss'] = (df['Low Sugar'] + df['Low Calorie'] + df['Low Fat'] >= 2).astype(int)
    df['Weight Gain'] = (df['High Calorie'] + df['High Protein'] >= 2).astype(int)
    df['Muscle Building'] = (df['High Protein'] + df['High Fat'] >= 2).astype(int)
    df['Blood Pressure Management'] = (df['Low Sodium'] + df['Moderate Sodium'] >= 1).astype(int)
    
    # Create combined health tags
    df['Health_tags'] = df.apply(combine_tags, axis=1)

    # Add columns for tag categories
    for category, tags in tag_categories.items():
        df[category] = df['tags'].apply(lambda x: [tag for tag in x if tag in tags])

    return df


# Function to combine derived health tags
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

# Function to calculate and return recommendations based on all tags
def get_recommendations_all_tags(df, recipe_index, top_n=5):
    # Combine all tags into a single string
    df['all_tags_combined'] = df.apply(
        lambda row: ' '.join(row['tags'] + [row['Health_tags']]), axis=1
    )
    # Vectorize the combined tags
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '))
    features_matrix = vectorizer.fit_transform(df['all_tags_combined'])
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(features_matrix)
    # Convert similarity matrix to DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)
    # Get top N similar recipes
    similar_scores = similarity_df[recipe_index].sort_values(ascending=False)
    similar_recipes = similar_scores.drop(recipe_index).head(top_n)
    return similar_recipes

def get_recommendations_for_group(recipes_df, tag_column, recipe_index, top_n=5):
    # Combine features for the specified tag column
    recipes_df[f'{tag_column}_features'] = recipes_df[tag_column].apply(
        lambda tags: ' '.join(tags) if isinstance(tags, list) else ''
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
    "specific_ingredients_tags": ['chocolate', 'berries', 'tropical-fruit', 'onions', 'citrus', 'potatoes', 'carrots', 'mushrooms'],
    # Add health tags
    "health_tags": [
        'Weight Loss', 'Weight Gain', 'Muscle Building', 'Blood Pressure Management',
        'High Protein', 'Very High Protein', 'Low Sugar', 'Low Fat', 'Low Calorie',
        'High Calorie', 'High Fat', 'Moderate Sodium', 'Low Sodium'
    ]
}

def preprocess_recipes(recipes_df):
    # Fit vectorizers on the entire dataset
    vectorizer_ingredients = TfidfVectorizer(max_features=1000, stop_words='english')
    vectorizer_tags = TfidfVectorizer(max_features=1000, stop_words='english')
    
    # Vectorize ingredients and health tags
    ingredients_tfidf = vectorizer_ingredients.fit_transform(recipes_df['ingredients'].astype(str))
    health_tags_tfidf = vectorizer_tags.fit_transform(recipes_df['Health_tags'].astype(str))
    
    # Normalize numerical columns
    scaler = MinMaxScaler()
    nutrition_features = scaler.fit_transform(recipes_df[['calories', 'protein (PDV)']])
    
    # Combine all features
    recipe_features = np.hstack((ingredients_tfidf.toarray(), health_tags_tfidf.toarray(), nutrition_features))
    return recipe_features, vectorizer_ingredients, vectorizer_tags



def build_user_profile(preferred_ingredients, goal, vectorizer_ingredients, vectorizer_tags):
    # Transform user preferences using the pre-fitted vectorizers
    preferred_vector = vectorizer_ingredients.transform([' '.join(preferred_ingredients)]).toarray()
    goal_vector = vectorizer_tags.transform([goal]).toarray()
    
    # Combine features into a single user profile vector
    user_profile = np.hstack((preferred_vector, goal_vector, [[0.5, 0.5]]))  # Add dummy nutritional features if needed
    return user_profile


def recommend_recipes(user_profile, recipe_features, recipes_df, top_n=1):
    # Compute cosine similarity between the user profile and recipe features
    similarity_scores = cosine_similarity(user_profile, recipe_features)
    recipes_df['similarity_score'] = similarity_scores[0]
    
    # Rank recipes by similarity score
    recommended_recipes = recipes_df.sort_values(by='similarity_score', ascending=False)
    return recommended_recipes.head(top_n)


def generate_daily_diet_plan_content_based(goal, preferred_ingredients):
    # Preprocess the entire dataset to get precomputed features and vectorizers
    recipe_features, vectorizer_ingredients, vectorizer_tags = preprocess_recipes(df)
    
    # Build the user profile
    user_profile = build_user_profile(preferred_ingredients, goal, vectorizer_ingredients, vectorizer_tags)
    course_tags = [
    'breakfast', 'lunch','snacks', 'dinner-party', 'desserts']
    # Generate the daily plan
    daily_plan = []
    for course in course_tags:
        print(f"Processing course: {course}...")
        
        # Filter recipes for the current course
        course_indices = df['course_tags'].apply(lambda x: course in x if isinstance(x, list) else False)
        course_data = df[course_indices]
        
        # Filter precomputed features for the current course
        course_features = recipe_features[course_indices.to_numpy()]
        
        if not course_data.empty:
            # Recommend the top recipes for the course
            recommended = recommend_recipes(user_profile, course_features, course_data, top_n=1)
            for _, recipe in recommended.iterrows():
                daily_plan.append({
                    'Course': course.capitalize(),
                    'Recipe Name': recipe['name'],
                    'Calories': recipe['calories'],
                    'Protein': recipe['protein (PDV)'],
                    'Ingredients': recipe['ingredients']
                })
    
    # Check if the plan is empty
    if daily_plan:
        print("Successfully generated the diet plan.")
    else:
        print("No recipes matched the criteria. Please adjust your inputs.")
    
    return pd.DataFrame(daily_plan)








# Load dataset
df = load_data()

# Streamlit UI
st.title("Multi-Level Recipe Recommendation System with Health Tags")

# User selects a recipe (clearable dropdown)
selected_recipe_name = st.selectbox(
    "Select a Recipe",
    options=[""] + df['name'].tolist(),  # Add an empty option for clearing
    format_func=lambda x: "Select a Recipe" if x == "" else x,  # Placeholder
)

if selected_recipe_name:
    # Get the index of the selected recipe
    selected_recipe_index = df[df['name'] == selected_recipe_name].index[0]
    st.subheader(f"Selected Recipe: {selected_recipe_name}")
    st.write(f"Tags: {df.loc[selected_recipe_index, 'tags']}")
    st.write(f"Health Tags: {df.loc[selected_recipe_index, 'Health_tags']}")

    # Show recommendations based on all tags
    st.subheader("Most Similar Recipes Based on Tags and Health Categories")
    recommendations = get_recommendations_all_tags(df, selected_recipe_index, top_n=5)
    for idx in recommendations.index:
        st.write(f"- **{df.loc[idx, 'name']}**")
        st.write(f"Tags: {df.loc[idx, 'tags']}")
        st.write(f"Health Tags: {df.loc[idx, 'Health_tags']}")

# Optional: User selects a tag group for more detailed recommendations
st.subheader("Detailed Recommendations by Tag Group")
selected_group = st.selectbox("Select Tag Group for Detailed Recommendations", list(tag_categories.keys()))

if st.button("Generate Detailed Recommendations"):
    try:
        # Generate recommendations for the selected tag group
        group_recommendations = get_recommendations_for_group(df, selected_group, selected_recipe_index, top_n=5)
        st.subheader(f"Recommendations Based on {selected_group.replace('_', ' ').capitalize()}:")
        for idx in group_recommendations.index:
            st.write(f"- **{df.loc[idx, 'name']}**")
            st.write(f"Tags: {df.loc[idx, selected_group]}")
            if selected_group == 'health_tags':
                st.write(f"Health Tags: {df.loc[idx, 'Health_tags']}")
    except ValueError as e:
        st.error(f"No valid features for {selected_group}: {e}")


st.subheader("Personalized Recipe Recommendation and Diet Plan")
#st.subheader("Enter Your Preferences")

ingredient_options = tag_categories["ingredient_tags"] + tag_categories["specific_ingredients_tags"]
selected_ingredients = st.multiselect(
    "Select Your Preferred Ingredients",
    options=ingredient_options,
    default=[]
)

# Dietary goal selection
goal = st.selectbox(
    "Select Your Dietary Goal",
    tag_categories["health_tags"]
)

# Generate diet plan button
if st.button("Generate Diet Plan"):
    # Load dataset
    df = load_data()
    
    # Generate diet plan
    diet_plan = generate_daily_diet_plan_content_based(goal, selected_ingredients)
    
    # Display diet plan
    if not diet_plan.empty:
        st.subheader("Your Personalized Daily Diet Plan")
        for _, row in diet_plan.iterrows():
            st.write(f"### {row['Course']}")
            st.write(f"**Recipe**: {row['Recipe Name']}")
            st.write(f"**Calories**: {row['Calories']:.2f}, **Protein**: {row['Protein']:.2f}")
            # Parse ingredients if they are in string format
            ingredients = row['Ingredients']
            if isinstance(ingredients, str):
                try:
                    ingredients = ast.literal_eval(ingredients)  # Convert to a list if needed
                except (ValueError, SyntaxError):
                    ingredients = [ingredients]  # Fallback to treat as a single ingredient

            # Display ingredients properly
            st.write(f"**Ingredients**: {', '.join(ingredients)}")
            st.write("---")
    else:
        st.warning("No matching recipes found for the selected goal and ingredients.")