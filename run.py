import json
import parser
import my_ai

classifier = my_ai.create_model(
        data_path='code_dataset.csv',
        model_type='nn',
        epochs=250
    )
a = """a = input()
if '@' in a and '.' in a:
    print("YES")
else:
    print("NO")
"""
code = '\\n '.join(a.split('\n'))
predictions = my_ai.check(classifier, code)

print(predictions)
