import json
import parser
import my_ai

classifier = my_ai.load(model_path="code_classifier")
a = """a = input()
print("HELLO:")
"""

b = """a = input()
if (a == '@' and a == '.'):
    print("YES")
else:
    print("NO")
"""
print(my_ai.compare(a, b))
