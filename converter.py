import pyperclip


a = """n = int(input())

for i in range(n):
    res =[]
    for j in range(n):
        if (i >= j and i >= n - 1 - j) or (i <= j and i <= n - 1 - j):
            res.append(1)
        else:
            res.append(0)
    print(*res)
"""

pyperclip.copy('\\n '.join(a.split('\n')))
