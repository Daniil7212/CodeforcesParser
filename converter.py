import pyperclip

a = """def getprimes(n):
    result = dict()
    i = 2
    while i * i <= n:
        if n % i == 0:
            if i not in result:
                result[i] = 0
            result[i] += 1
            n //= i
        else:
            i += 1
    
    if n > 1:
        if n in result:
            result[n] += 1
        else:
            result[n] = 1

    return result

n = int(input())
print("*".join([(str(i[0]) + "^" + str(i[1]) if i[1] > 1 else str(i[0])) for i in sorted(getprimes(n).items())]))
"""

pyperclip.copy('\\n '.join(a.split('\n')))
