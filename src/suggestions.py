from src.family import load_families_from_json, extract_all_photo_paths_from_json
from src.embedding_generation import get_files_faces
from src.short_list_selection import select_closest
from src.classification_utils import (make_classification_tasks,
                                      make_classification_df,
                                      predict_probabilities,
                                      get_top_suggestions,
                                      filter_suggestions)

import pickle


def prepare_families(json_path,
                     embeddings_path=None):

    photo_paths = extract_all_photo_paths_from_json(json_path)

    if embeddings_path is None:
        embeddings = get_files_faces(photo_paths) #with extra arguments if necessary
    else:
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)

    families = load_families_from_json(json_path, embeddings)

    return families

def list_families(families):
    families_set = set(families.keys())
    return list(families_set)


def make_suggestions_df(families, family_id, model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    family = families[family_id]

    people = []
    for el in families.values():
        if el != family:
            people += list(el.members.values())
    closest_people_found = select_closest(family, people, n=10)
    short_list = [el[0] for el in closest_people_found]

    tasks = make_classification_tasks(family, short_list)
    tasks_df = make_classification_df(tasks)
    tasks_df = predict_probabilities(tasks_df, model, non_feature_cols=[])
    top_suggestions = get_top_suggestions(tasks_df, answers_known=False)
    top_suggestions = filter_suggestions(top_suggestions, families)
    return top_suggestions
