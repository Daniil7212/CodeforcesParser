import json
import parser

CONTEST_N = 9
IS_IN_TIME = True
LOGIN = "sirius-0625-079"
PASSWORD = "bitutobab"

with open('deepseek_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

data = data['students']



