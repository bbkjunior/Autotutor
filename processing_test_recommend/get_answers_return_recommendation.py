from get_marked_datasets import generate_user_knowledge_database
from knn_sklearn_recommendation import get_recommended_text_json

import argparse
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('answers_json', help='path to the file with raw text')
args = parser.parse_args()


generate_user_knowledge_database(args.answers_json)
get_recommended_text_json(args.answers_json)