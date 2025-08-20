import json
import parser
import my_ai

classifier = my_ai.load_model()

a = """n=int(input())
flag = 0

if n%4==0 and n%3!=0:
    print(4)
else:
    while n%2==0:
        n=n//2

    flag = 0
    for i in range(3, int(n ** 0.5) + 1):
        if n % i == 0:
            print(i)
            flag = 1
            break


    if flag==0 and n!=1:
        print(n)


"""

print(my_ai.check(classifier, a))
