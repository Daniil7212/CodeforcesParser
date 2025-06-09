import pyperclip


a = """def check(m,a,k):
    kol=0
    for i in range (len(a)):
        kol+=a[i]//m
    if kol>=k:
        return True
    else:
        return False
n,k=map(int,input().split())
a=[]
for i in range (n):
    x=int(input())
    a.append(x)
l=0
r=1000000000000
while r-l>1:
    m=(r+l)//2
    if check(m,a,k):
        l=m
    else:
        r=m
print(l)
"""

pyperclip.copy('\\n '.join(a.split('\n')))
