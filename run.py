import json
import parser
import my_ai

classifier = my_ai.load_model()

a = """t = 0
print(t)
"""

print(my_ai.check(classifier, a))
