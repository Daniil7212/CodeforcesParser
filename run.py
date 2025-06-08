import json
import parser
import my_ai

classifier = my_ai.create_model(
        data_path='code_dataset.csv',
        model_type='nn',
        epochs=250
    )
a = """
n = int(input())
sm = 0
a = list(map(int, input().split()))
a.sort()
for i in range(n - 1):
    lf = i - 1
    r = len(a)
    while lf + 1 < r:
        m = (lf + r) // 2
        if a[i] < a[m] * 0.9:
            r = m
        else:
            lf = m
    sm += r - i - 1
print(sm)
"""
code = '\\n '.join(a.split('\n'))
predictions = my_ai.check(classifier, code)

print(predictions)
