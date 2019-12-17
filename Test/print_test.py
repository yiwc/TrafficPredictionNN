# with open('out.txt', 'w') as f:
f=open('out.txt', 'w+')
print('Filename:', "testcont", file=f)
print('Filename:', "testcont2", file=f)
print('Filename:', "testcont3", file=f)
f.close()