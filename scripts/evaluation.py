def evaluate_recommendations_with_best_score(recommended_recipe_ids, liked_recipe_ids, similarity_scores):
    """
    Evaluate the relevance of recommendations by comparing with liked recipes
    and calculate the best match score.

    Args:
        recommended_recipe_ids (list): List of recipe IDs recommended.
        liked_recipe_ids (list): List of recipe IDs liked by users.
        similarity_scores (list): List of similarity scores corresponding to the recommendations.

    Returns:
        dict: Dictionary containing evaluation metrics (matches, precision, recall, best_match_score).
    """
    # Convert to sets for efficient comparison
    recommended_set = set(recommended_recipe_ids)
    liked_set = set(liked_recipe_ids)

    # Find matches
    matches = recommended_set & liked_set
    num_matches = len(matches)

    # Precision: Fraction of recommended recipes that are relevant
    precision = num_matches / len(recommended_recipe_ids) if len(recommended_recipe_ids) > 0 else 0

    # Recall: Fraction of relevant recipes that were recommended
    recall = num_matches / len(liked_recipe_ids) if len(liked_recipe_ids) > 0 else 0

    # Calculate best match score
    best_match_score = 0
    if num_matches > 0:
        match_indices = [i for i, recipe_id in enumerate(recommended_recipe_ids) if recipe_id in matches]
        weighted_similarities = sum([similarity_scores[i] for i in match_indices])
        best_match_score = weighted_similarities / num_matches  # Average match score

    return {
        "matches": matches,
        "num_matches": num_matches,
        "precision": precision,
        "recall": recall,
        "best_match_score": best_match_score
    }
