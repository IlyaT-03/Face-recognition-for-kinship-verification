from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import date
import json
from collections import defaultdict
import numpy as np
from copy import deepcopy


@dataclass
class Person:
    name: Optional[str] = None
    surname: Optional[str] = None
    middleName: Optional[str] = None
    birthplace: Optional[str]= None
    birthdate: Optional[date] = None
    gender: Optional[str] = None
    family_id: Optional[str] = None
    person_id: Optional[str] = None
    embeddings: List[np.ndarray] = field(default_factory=list)
    photo_paths: List[str] = field(default_factory=list)
    mean_embedding: Optional[np.ndarray] = None
    mother: List['Person'] = field(default_factory=list)
    father: List['Person'] = field(default_factory=list)
    brothers: List['Person'] = field(default_factory=list)
    sisters: List['Person'] = field(default_factory=list)
    daughters: List['Person'] = field(default_factory=list)
    sons: List['Person'] = field(default_factory=list)
    spouses: List['Person'] = field(default_factory=list)

    # --- Grandparents ---
    def get_grandmothers(self) -> List['Person']:
        return [g for p in self.mother + self.father for g in p.mother]

    def get_grandfathers(self) -> List['Person']:
        return [g for p in self.mother + self.father for g in p.father]

    # --- Grandchildren ---
    def get_granddaughters(self) -> List['Person']:
        return [g for c in self.daughters + self.sons for g in c.daughters]

    def get_grandsons(self) -> List['Person']:
        return [g for c in self.daughters + self.sons for g in c.sons]

    # --- Aunts and Uncles ---
    def get_aunts(self) -> List['Person']:
        return [a for p in self.mother + self.father for a in p.sisters]

    def get_uncles(self) -> List['Person']:
        return [u for p in self.mother + self.father for u in p.brothers]

    # --- Nieces and Nephews ---
    def get_nieces(self) -> List['Person']:
        return [n for s in self.sisters + self.brothers for n in s.daughters]

    def get_nephews(self) -> List['Person']:
        return [n for s in self.sisters + self.brothers for n in s.sons]


@dataclass
class Family:
    members: Dict[str, Person] = field(default_factory=dict)
    family_id: Optional[str] = None

    def create_family_excluding_a_person(self, exclude_id: str) -> 'Family':
        # Deep copy the entire family
        new_family = deepcopy(self)

        # Remove the excluded person
        if exclude_id in new_family.members:
            del new_family.members[exclude_id]

        # Remove all links to the excluded person in others
        for person in new_family.members.values():
            for rel_attr in ["mother", "father", "brothers", "sisters", "daughters", "sons", "spouses"]:
                relatives = getattr(person, rel_attr)
                filtered = [p for p in relatives if p.person_id != exclude_id]
                setattr(person, rel_attr, filtered)

        return new_family


def parse_birthdate(b):
    if not b or not all(k in b for k in ("year", "month", "day")):
        return None
    try:
        return date(b["year"], b["month"], b["day"])
    except Exception:
        return None


def extract_all_photo_paths_from_json(path: str) -> List[str]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    photo_paths = set()
    for person in data:
        for photo in person.get("photo_paths", []):
            photo_paths.add(photo)

    return list(photo_paths)


def load_families_from_json(
    path: str,
    embeddings_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None
) -> Dict[str, Family]:
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 1. Создание всех Person
    all_people: Dict[str, Person] = {}
    for person_dict in raw_data:
        photo_paths = person_dict.get("photo_paths", [])

        person = Person(
            person_id=person_dict["person_id"],
            family_id=person_dict["family_id"],
            name=person_dict.get("name"),
            surname=person_dict.get("surname"),
            middleName=person_dict.get("middleName"),
            birthplace=person_dict.get("birthplace"),
            gender=person_dict.get("gender"),
            birthdate=parse_birthdate(person_dict.get("birthdate")),
            photo_paths=photo_paths,
        )

        # Используем embeddings_dict, если есть
        if embeddings_dict:
            for path in photo_paths:
                emb_info = embeddings_dict.get(path)
                if emb_info and "embedding" in emb_info:
                    person.embeddings.append(emb_info["embedding"])
            if person.embeddings:
                person.mean_embedding = np.mean(person.embeddings, axis=0)

        all_people[person.person_id] = person

    # 2. Связи между родственниками
    for person_dict in raw_data:
        source = all_people[person_dict["person_id"]]
        for rel in person_dict.get("relatives", []):
            target = all_people.get(rel["person_id"])
            if not target:
                continue
            rel_type = rel["relationType"]
            gender = (target.gender or "").lower()

            if rel_type == "parent":
                if gender == "female":
                    source.mother.append(target)
                elif gender == "male":
                    source.father.append(target)
            elif rel_type == "child":
                if gender == "female":
                    source.daughters.append(target)
                elif gender == "male":
                    source.sons.append(target)
            elif rel_type == "spouse":
                source.spouses.append(target)

    # 2.5 Братья и сестры
    for person in all_people.values():
        sibling_ids = set()
        for parent in person.mother + person.father:
            for child in parent.daughters + parent.sons:
                if child.person_id != person.person_id:
                    sibling_ids.add(child.person_id)

        for sid in sibling_ids:
            sibling = all_people[sid]
            gender = (sibling.gender or "").lower()
            if gender == "male":
                person.brothers.append(sibling)
            elif gender == "female":
                person.sisters.append(sibling)

    # 3. Группировка по семьям
    families: Dict[str, Family] = defaultdict(lambda: Family(members={}, family_id=None))
    for person in all_people.values():
        fam_id = person.family_id
        families[fam_id].family_id = fam_id
        families[fam_id].members[person.person_id] = person

    return families

