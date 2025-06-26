from scipy.spatial.distance import cosine
import numpy as np


def distance_to_other_person_relatives(person_to_assign, people):
    other_person_relatives_embeddings = []
    embedding_to_assign = person_to_assign.mean_embedding
    if embedding_to_assign is None:
        return None
    for person in people:
        embedding = person.mean_embedding
        if person.person_id != person_to_assign.person_id and embedding is not None:
            other_person_relatives_embeddings.append(embedding)
    if len(other_person_relatives_embeddings) > 0:
        other_person_relatives_embedding = np.mean(other_person_relatives_embeddings, axis=0)
        return cosine(embedding_to_assign, other_person_relatives_embedding)
    return None


def birthplace_distance(birthplace1, birthplace2):
    return None

def surname_distance(surname1, surname2):
    return None

def birthdate_distance(birthdate1, birthdate2):
    return None

def name_middlename_distance(name, middlename):
    return None


def calculate_features(person_to_assign, other_person):

    features = {}

    # --- Расстояние по месту рождения (заглушка) ---
    features["birthplace_distance"] = birthplace_distance(
        person_to_assign.birthplace,
        other_person.birthplace
    )

    # --- Расстояние по фамилии (заглушка) ---
    features["surname_distance"] = surname_distance(
        person_to_assign.surname,
        other_person.surname
    )

    # --- Расстояние по дате рождения (заглушка) ---
    features["birthdate_distance"] = birthdate_distance(
        person_to_assign.birthdate,
        other_person.birthdate
    )

    # --- Совпадение имени и отчества (заглушка) ---
    features["person_to_assign_name_other_person_middlename_distance"] = name_middlename_distance(
        person_to_assign.name,
        other_person.middleName
    )
    features["other_person_name_person_to_assign_middlename_distance"] = name_middlename_distance(
        other_person.name,
        person_to_assign.middleName
    )

    features["gender_male"] = person_to_assign.gender == 'Male'
    features["other_person_gender_male"] = other_person.gender == 'Male'

    embedding = person_to_assign.mean_embedding
    other_embedding = other_person.mean_embedding
    if embedding is None or other_embedding is None:
        distance = None
    else:
        distance = cosine(embedding, other_person.mean_embedding)
    features["distance"] = distance

    # --- Родственники ---
    features["sons_distance"] = distance_to_other_person_relatives(person_to_assign, other_person.sons)
    features["daughters_distance"] = distance_to_other_person_relatives(person_to_assign, other_person.daughters)
    features["sisters_distance"] = distance_to_other_person_relatives(person_to_assign, other_person.sisters)
    features["brothers_distance"] = distance_to_other_person_relatives(person_to_assign, other_person.brothers)
    features["father_distance"] = distance_to_other_person_relatives(person_to_assign, other_person.father)
    features["mother_distance"] = distance_to_other_person_relatives(person_to_assign, other_person.mother)
    features["spouses_distance"] = distance_to_other_person_relatives(person_to_assign, other_person.spouses)

    # --- Расширенные родственники ---
    features["grandmothers_distance"] = distance_to_other_person_relatives(person_to_assign,
                                                                           other_person.get_grandmothers())
    features["grandfathers_distance"] = distance_to_other_person_relatives(person_to_assign,
                                                                           other_person.get_grandfathers())
    features["granddaughters_distance"] = distance_to_other_person_relatives(person_to_assign,
                                                                             other_person.get_granddaughters())
    features["grandsons_distance"] = distance_to_other_person_relatives(person_to_assign, other_person.get_grandsons())
    features["nieces_distance"] = distance_to_other_person_relatives(person_to_assign, other_person.get_nieces())
    features["nephews_distance"] = distance_to_other_person_relatives(person_to_assign, other_person.get_nephews())
    features["aunts_distance"] = distance_to_other_person_relatives(person_to_assign, other_person.get_aunts())
    features["uncles_distance"] = distance_to_other_person_relatives(person_to_assign, other_person.get_uncles())

    return features
