import re
from typing import List, Any, Dict

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

stop_words_set = set(stopwords.words('english'))

EN_PREFIX = "/c/en/"
EN_CONCEPT_RE = re.compile("/c/en/([^/]*).*")

def convert_entity_to_string(ent, ent2label):
    # clean entity name if needed
	if ent in ent2label.keys():
		return ent2label[ent]
	else:
		return ''

def load_node_labels(nodes_path):
	node2label={}
	with open(nodes_path, 'r') as nodes_f:
		for line in nodes_f:
			node_id, label, *rest = line.split('\t')
			node2label[node_id]=label
	return node2label

def tokenize_str(input_str):
    return [stemmer.stem(str) for str in re.split("[\W_]+", input_str.lower())
            if str not in stop_words_set]


def tokenize_and_stem_str(input_str):
    return [(str, stemmer.stem(str)) for str in re.split("[\W_]+", input_str.lower())
            if str not in stop_words_set]


def accept_relation(rel, ignore_related=False):
    if rel == "/r/Antonym":
        return False
    if rel.startswith("/r/dbpedia/"):
        return False
    if rel.startswith("/r/Etymologically"):
        return False
    if ignore_related and rel.startswith("/r/RelatedTo"):
        return False
    if rel=='vg:InImage':
        return False
    return True


def convert_relation_to_string(rel):
    if rel.lower() == "/r/isa":
        return "a type of"
    if rel.lower() == "/r/none":
        return "not related to"
    return " ".join([x.lower() for x in split_relation(rel)])


def split_relation(rel):
    if rel.startswith("/r/"):
        rel = rel[len("/r/"):]
    else:
        rel=rel.split(':')[-1]
    return re.split("[\W_]+", re.sub('([a-z])([A-Z])', r'\1 \2', rel))

def load_kbtuples_map(kg_path, node2label, ignore_related=False) -> (List[Any], Dict[str, List[int]]):
    kg_triples = []
    kg_map = {}
    with open(kg_path, 'r') as kg_file:
        for line in kg_file:
            field = line.strip().split("\t")
            ent1 = convert_entity_to_string(field[0], node2label)
            rel = field[1]
            ent2 = convert_entity_to_string(field[2], node2label)
            if ent1 and ent2 and accept_relation(rel, ignore_related):
                ent1_toks = tokenize_str(ent1)
                ent2_toks = tokenize_str(ent2)
                kg_triples.append((ent1, rel, ent2))
                for ent1_tok in ent1_toks:
                    if ent1_tok not in kg_map:
                        kg_map[ent1_tok] = []
                    kg_map[ent1_tok].append(len(kg_triples) - 1)
                for ent2_tok in ent2_toks:
                    if ent2_tok not in kg_map:
                        kg_map[ent2_tok] = []
                    kg_map[ent2_tok].append(len(kg_triples) - 1)
    return kg_triples, kg_map


def retrieve_scored_tuples(ent1, ent2, kbtuples, kbmap, max=100):
    match_set = set()
    num_relations = 0
    ent1_toks = tokenize_and_stem_str(ent1)
    ent2_toks = tokenize_and_stem_str(ent2)
    additional_tuples = []
    for (ent1_orig, ent1_tok) in ent1_toks:
        ent1_set = set(kbmap.get(ent1_tok, []))
        for (ent2_orig, ent2_tok) in ent2_toks:
            match_found = False
            if ent2_tok == ent1_tok:
                additional_tuples.append((ent1_orig, "/r/SameAs", ent2_orig))
                num_relations += 1
                continue
            ent2_set = set(kbmap.get(ent2_tok, []))
            ent1_ent2_inter = ent1_set.intersection(ent2_set)
            for idx in ent1_ent2_inter:
                kb_ent1_toks = tokenize_str(kbtuples[idx][0])
                kb_ent2_toks = tokenize_str(kbtuples[idx][2])
                if ent1_tok in kb_ent1_toks and ent2_tok in kb_ent2_toks:
                    match_set.add(idx)
                    match_found = True
                elif ent1_tok in kb_ent2_toks and ent2_tok in kb_ent1_toks:
                    match_set.add(idx)
                    match_found = True
            if match_found:
                num_relations += 1
    scored_tuples = []
    for tuple in additional_tuples:
        scored_tuples.append((tuple, 1.0))
    if not len(scored_tuples) and not len(match_set):
        return [((ent1, "/r/NONE", ent2), 0.0)]
    for tupleidx in match_set:
        ent_toks = set([x[1] for x in ent1_toks] + [x[1] for x in ent2_toks])
        kb_toks = set(tokenize_str(kbtuples[tupleidx][0]) + tokenize_str(kbtuples[tupleidx][2]))
        if len(ent_toks) or len(kb_toks):
            score = len(ent_toks.intersection(kb_toks)) / len(ent_toks.union(kb_toks))
            scored_tuples.append((kbtuples[tupleidx], score))
    scored_tuples.sort(key=lambda x: -x[1])
    return scored_tuples[:max]
