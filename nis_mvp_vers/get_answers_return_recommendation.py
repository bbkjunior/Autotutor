from get_marked_datasets import generate_user_knowledge_database
from knn_sklearn_recommendation import get_recommended_text_json
from collections import OrderedDict
"""
import argparse
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('answers_json', help='path to the file with raw text')
args = parser.parse_args()
"""

answer_dict = {
    "user_id":"pupkin_zalupkin",
    "answers": {0: OrderedDict([(0, True), (2, False), (5, False)]),
    17: OrderedDict([(0, False),
                 (2, False),
                 (4, False),
                 (8, True),
                 (10, False),
                 (12, False)]),
    26: OrderedDict([(0, False), (5, True), (7, True), (13, False)]),
    32: OrderedDict([(3, True), (5, True), (14, False), (18, False)]),
    98: OrderedDict([(0, True), (4, True), (5, True), (8, False)]),
    121: OrderedDict([#(0, True),вопрос проебан просто не вставлен в тест!!!!
                 (1, True),
                 (3, False),
                 (4, True),
                 (6, False),
                 (11, True)]),
    130: OrderedDict([(0, True),
                 (1, False),
                # (3, True),вопрос проебан просто не вставлен в тест!!!!
                 (4, False),
                 (5, False),
                 (8, False),
                 (9, True),
                 (16, True)]),
    133: OrderedDict([(2, False),
                 (6, False),
                 (8, False),
                 (9, False),
                 (18, False),
                 (20, False),
                 (40, True),
                 (49, False)]),
    200: OrderedDict([(0, False),(2, False), (4, False), (10, True), (12, True)]),
    231: OrderedDict([(2, False), (4, True), (7, False), (12, False)]),
    240: OrderedDict([(1, True), (11, False), (15, False)]),
    316: OrderedDict([(0, True), (5, False), (7, False), (11, True)]),
    331: OrderedDict([(2, False), (4, False), (10, False)]),
    334: OrderedDict([(1, False), (2, False), (5, True), (8, False)]),
    336: OrderedDict([(1, True), (7, True), (10, True), (13, True)]),
    366: OrderedDict([(0, True),
                 (1, False),
                 (5, False),
                 (8, False),
                 (10, True),
                 (11, True),
                 (15, False)]),
    371: OrderedDict([(0, False),
                 (2, False),
                 (3, False),
                 (6, False),
                 (12, False),
                 (21, False)])}
                 }
def recommendation_algo(answers_json):
    #generate_user_knowledge_database(answers_json)
    recommendations = get_recommended_text_json(answers_json, save_json_to_directory = True)
    return recommendations

recommendation_algo(answer_dict)