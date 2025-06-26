kinship_dictionary = {
    "parent_probability": "parent",
    "child_probability": "child",
    "sibling_probability": "sibling"
}


def print_suggestions(suggestions_df, threshold=0.1):
    for ind in suggestions_df.index:
        row = suggestions_df.loc[ind]
        index = row['df_index']
        kinship_category = kinship_dictionary[row['score_type']]
        probability = round(row['probability'], 3)
        person_found = index[0]
        relative = index[2]
        if probability >= threshold:
            print(f"{person_found} is a {kinship_category} of {relative}\t - probability {probability}")
