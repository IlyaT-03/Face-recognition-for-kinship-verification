import pandas as pd
from src.feature_calculation import calculate_features
import numpy as np


def exclude_additional(family):
    for person in family.members.values():
        person_id = person.person_id
        if 'additional' in person_id:
            family = family.create_family_excluding_a_person(person_id)
    return family



def make_classification_tasks_with_known_answers(family,
                                                 people_to_assign,
                                                 excluding_additional=True):
    if excluding_additional:
        family = exclude_additional(family)
    family_id = family.family_id
    classification_tasks = []
    for person_to_assign in people_to_assign:
        for family_person in family.members.values():
            if family_person.person_id not in [el.person_id for el in people_to_assign]:

                if person_to_assign.family_id != family_id:
                    relation_type = 'other'
                elif family_person != person_to_assign:
                    family_person_parents = family_person.mother + family_person.father
                    family_person_children = family_person.sons + family_person.daughters
                    family_person_siblings = family_person.sisters + family_person.brothers
                    if person_to_assign.person_id in [p.person_id for p in family_person_parents]:
                        relation_type = 'parent'
                    elif person_to_assign.person_id in [p.person_id for p in family_person_children]:
                        relation_type = 'child'
                    elif person_to_assign.person_id in [p.person_id for p in family_person_siblings]:
                        relation_type = 'sibling'
                    else:
                        relation_type = 'other'

                classification_tasks.append((person_to_assign, family_person, relation_type))

    return classification_tasks


def make_classification_tasks(family,
                              people_to_assign,
                              excluding_additional=True):
    if excluding_additional:
        family = exclude_additional(family)
    classification_tasks = []
    for person_to_assign in people_to_assign:
        for family_member in family.members.values():
            classification_tasks.append((person_to_assign, family_member))
    return classification_tasks


def make_classification_df(classification_tasks, answers_known=False):
    index = []
    rows = []
    for classification_task in classification_tasks:
        person_to_assign = classification_task[0]
        family_person = classification_task[1]

        person_to_assign_id = person_to_assign.person_id
        person_to_assign_family_id = person_to_assign.family_id

        family_person_id = family_person.person_id
        family_id = family_person.family_id

        index.append((person_to_assign_id, person_to_assign_family_id, family_person_id, family_id))

        features = calculate_features(person_to_assign, family_person)

        if answers_known:
            kinship = classification_task[2]
            features["kinship"] = kinship

        rows.append(features)

    df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index,
                                                            names=["person_to_assign_id", "person_to_assign_family_id",
                                                                   "family_person_id", "family_id"]))
    return df



def predict_probabilities(df,
                          model,
                          non_feature_cols = []):

    df = df.copy()
    classes = model.classes_
    not_relative_index = int(np.where(classes == 'other')[0][0])
    child_index = int(np.where(classes == 'child')[0][0])
    parent_index = int(np.where(classes == 'parent')[0][0])
    sibling_index = int(np.where(classes == 'sibling')[0][0])

    X = df.drop(non_feature_cols, axis=1)
    probs = model.predict_proba(X)

    df["not_relative_probability"] = probs[:, not_relative_index]
    df["child_probability"] = probs[:, child_index]
    df["parent_probability"] = probs[:, parent_index]
    df["sibling_probability"] = probs[:, sibling_index]

    return df


def get_top_suggestions(df, answers_known=True):
    df = df.copy()
    df['df_index'] = df.index
    if answers_known:
        df = df[['child_probability', 'parent_probability', 'sibling_probability', 'kinship', 'not_relative_probability',
                 'df_index']]
        id_variables = ['kinship', 'not_relative_probability', 'df_index']
    else:
        df = df[
            ['child_probability', 'parent_probability', 'sibling_probability', 'not_relative_probability',
             'df_index']]
        id_variables = ['not_relative_probability', 'df_index']
    df = df.melt(id_vars=id_variables,
                 value_vars=['child_probability', 'parent_probability', 'sibling_probability'],
                 var_name="score_type",
                 value_name="probability")

    df = df.sort_values(by=['probability'], ascending=False)

    return df


def filter_suggestions(df, families):
    df = df.copy()
    for index in df.index:
        row = df.loc[index]
        #kinship = row['kinship']
        idx = row['df_index']
        score_type = row['score_type']

        family_id = idx[3]

        if score_type == 'parent_probability':
            person = families[family_id].members[idx[2]]
            person_to_assign = families[idx[1]].members[idx[0]]
            if person_to_assign.gender == "Male" and len(person.father) > 0 and not person_to_assign in person.father:
                df.loc[index, "possible"] = False
            elif person_to_assign.gender == "Female" and len(
                    person.mother) > 0 and not person_to_assign in person.mother:
                df.loc[index, "possible"] = False
            else:
                df.loc[index, "possible"] = True
        else:
            df.loc[index, "possible"] = True

    df = df[df['possible']]
    return df


def get_correctness(df):
    df = df.copy()
    for index in df.index:
        row = df.loc[index]
        kinship = row['kinship']
        score_type = row['score_type']
        idx = row['df_index']
        family_id = idx[3]
        correct_relative = (kinship == 'sibling' and score_type == 'sibling_probability') or (
                    kinship == 'parent' and score_type == 'parent_probability') or (
                                       kinship == 'child' and score_type == 'child_probability')
        correct_family = idx[1] == family_id
        correct_guess = correct_relative and correct_family

        df.loc[index, 'correct_guess'] = correct_guess
        df.loc[index, 'correct_family'] = correct_family

    return df


def evaluate_top_metrics(df):

    top_1 = df.iloc[0]["correct_guess"]
    top_3 = df.iloc[:3]["correct_guess"].any()
    top_5 = df.iloc[:5]["correct_guess"].any()

    top_1_family = df.iloc[0]["correct_family"]
    top_3_family = df.iloc[:3]["correct_family"].any()
    top_5_family = df.iloc[:5]["correct_family"].any()

    return top_1, top_3, top_5, top_1_family, top_3_family, top_5_family

