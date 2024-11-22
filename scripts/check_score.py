import pandas as pd

# Load the dataset
file_path = 'recommendation_evaluation_results_top25_100.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Create a 'HIT' column: 1 if Precision > 0, otherwise 0
df['HIT'] = (df['Precision'] > 0).astype(int)

# Group by tag group and calculate aggregates
tag_summary = df.groupby('Tag Group').agg(
    Average_Precision=('Precision', 'mean'),
    Total_Hits=('HIT', 'sum'),
    Total_Recommendations=('HIT', 'count')
).reset_index()

# Calculate HIT Score as percentage
tag_summary['HIT_Score'] = (tag_summary['Total_Hits'] / tag_summary['Total_Recommendations']) * 100

# Find the tag group with maximum and minimum precision
max_precision_tag = tag_summary.loc[tag_summary['Average_Precision'].idxmax()]
min_precision_tag = tag_summary.loc[tag_summary['Average_Precision'].idxmin()]

# Find the tag group with maximum and minimum HIT Score
max_hit_score_tag = tag_summary.loc[tag_summary['HIT_Score'].idxmax()]
min_hit_score_tag = tag_summary.loc[tag_summary['HIT_Score'].idxmin()]

# Display results
print(f"Tag Group with Maximum Precision: {max_precision_tag['Tag Group']}")
print(f"Maximum Precision Score: {max_precision_tag['Average_Precision']:.4f}")

print(f"\nTag Group with Minimum Precision: {min_precision_tag['Tag Group']}")
print(f"Minimum Precision Score: {min_precision_tag['Average_Precision']:.4f}")

print(f"\nTag Group with Maximum HIT Score: {max_hit_score_tag['Tag Group']}")
print(f"Maximum HIT Score: {max_hit_score_tag['HIT_Score']:.2f}%")

print(f"\nTag Group with Minimum HIT Score: {min_hit_score_tag['Tag Group']}")
print(f"Minimum HIT Score: {min_hit_score_tag['HIT_Score']:.2f}%")

# Save results to CSV for reference
output_file = 'Tag_Group_Summary_with_HIT_Scores.csv'
tag_summary.to_csv(output_file, index=False)
print(f"\nTag group summary with HIT scores saved to {output_file}")

# Optional: Display all tag precision and HIT scores
print("\nTag Group Summary with Precision and HIT Scores:")
print(tag_summary)
