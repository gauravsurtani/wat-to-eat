# Multi-Level Recipe Recommendation System

This project is a **Streamlit-based application** that provides **multi-level, tag-based recipe recommendations**. It processes a dataset of recipes with tags (e.g., cuisine, preparation time, dietary preferences) and recommends recipes similar to the one selected by the user. Recommendations are based on the cosine similarity of features derived from specific tag groups.

---

## Features

- **Tag-based Recommendations**:
  - Generates recommendations based on selected tag categories such as:
    - Time (e.g., "60-minutes-or-less")
    - Cuisine (e.g., "Mexican", "Italian")
    - Dietary Restrictions (e.g., "Vegetarian", "Low-carb")
    - Ingredients, Taste & Mood, Occasion, Technique, etc.
- **Dynamic Tag Group Selection**:
  - Allows users to select a recipe and tag group to generate targeted recommendations.
- **Efficient Data Processing**:
  - Processes large datasets efficiently with caching and subset-based loading for faster performance.

---

## How It Works

1. **Dataset Preparation**:
   - The `RAW_recipes.csv` file contains recipes with associated metadata, including a `tags` column.
   - The `tags` column is converted into categorized columns (e.g., `time_tags`, `cuisine_tags`) based on predefined tag categories.

2. **Feature Extraction**:
   - Features are combined for each tag group and vectorized using `CountVectorizer`.
   - Cosine similarity is calculated between recipes for the selected tag group.

3. **User Interaction**:
   - The user selects a recipe from a dropdown.
   - The user chooses a tag group for recommendations.
   - The app displays recipes most similar to the selected one within the chosen tag group.

---

## Setup and Installation

### Prerequisites
- Python 3.9 or later, I used Python 3.12.7 exactly
- Required libraries:
  - `streamlit`
  - `pandas`
  - `scikit-learn`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gauravsurtani/wat-to-eat.git
   cd multi-level-recipe-recommender
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the `RAW_recipes.csv` file in the root directory of the project.

---

## Usage

### Running the App

1. Start the Streamlit app:
   ```bash
   streamlit run main-v3.py
   ```

2. Open the app in your browser (typically `http://localhost:8501`).

### How to Use

1. **Select a Recipe**:
   - Use the dropdown to select a recipe from the dataset.

2. **Choose a Tag Group**:
   - Select a tag group (e.g., `time_tags`, `cuisine_tags`) from the second dropdown.

3. **Generate Recommendations**:
   - Click the "Generate Recommendations" button to display the most similar recipes.

4. **View Results**:
   - Recommended recipes will appear below, showing the recipe name and relevant tags.

#### Personalized Recipe Recommendation System:

1. **Select Preferred Ingredients**:
   - Choose your preferred ingredients from the available options.

2. **Select Dietary Goal**:
   - Pick a dietary goal (e.g., Weight Gain, Muscle Building) to align recommendations with your health objectives.

3. **Generate Personalized Diet Plan**:
   - Click the "Generate Diet Plan" button to create a personalized daily diet plan. The plan will include suggestions for breakfast, lunch, snacks, dinner, and desserts.


---

## File Structure

```plaintext
.
├── main-v3.py              # Main Streamlit application file
├── RAW_recipes.csv         # Recipe dataset
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation
```

---

## Example Walkthrough

1. Start the app and select a recipe, such as "Spicy Mexican Rice".
2. Choose a tag group like `cuisine_tags`.
3. Click "Generate Recommendations".
4. View recommended recipes like "Southwestern Rice Bowl" or "Mexican Fiesta Salad".

---

## Technologies Used

- **Streamlit**: For creating the web-based UI.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For vectorization and cosine similarity calculations.

---

## Notes

- Ensure that the `tags` column in the dataset is formatted as lists.
- If the dataset contains stringified lists, the application will automatically convert them to Python lists.
- For large datasets, only the first 10,000 rows are processed for performance optimization.

---

## Troubleshooting

- **Malformed Node or String**:
  - This error occurs if the `tags` column is not properly formatted. Ensure all tag values are lists or stringified lists.

- **Performance Issues**:
  - Reduce the number of rows processed by modifying the `n_rows` parameter in the `load_data` function.
