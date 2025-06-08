import json
import parser

CONTEST_N = 6
IS_IN_TIME = True
LOGIN = "sirius-0625-073"
PASSWORD = "womatowad"

with open('deepseek_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

data = data['students']

parser.parser(CONTEST_N, IS_IN_TIME, LOGIN, PASSWORD)

