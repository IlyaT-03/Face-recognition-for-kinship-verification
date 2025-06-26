from scipy.spatial.distance import cosine

def get_distance(a, b):
    if a is None or b is None:
        return None
    return cosine(a, b)


def get_top_unique(people_list, n=10):
    seen = set()
    result = []

    # Sort by the first element (ascending)
    for el in sorted(people_list, key=lambda x: x[1]):
        if el[2] not in seen:
            result.append(el)
            seen.add(el[2])
        if len(result) == n:
            break
    return result


def select_closest(family, people, n=10):
    distances = []
    for person in people:
        person_id = person.person_id
        person_embedding = person.mean_embedding
        if person_embedding is None:
            continue
        for family_member in family.members.values():
            embedding = family_member.mean_embedding
            if embedding is None:
                continue
            distance = get_distance(person_embedding, embedding)
            if distance is not None:
                distances.append((person, distance, person_id, family_member.person_id, person))
            for spouse in family_member.spouses:
                if spouse.mean_embedding is not None:
                    spouses_embedding = (spouse.mean_embedding + embedding) / 2
                    distance = get_distance(person_embedding, spouses_embedding)
                    if distance is not None:
                        distances.append((person, distance, person_id, (family_member.person_id, spouse.person_id)))

    people_to_return = get_top_unique(distances, n=n)
    return people_to_return
