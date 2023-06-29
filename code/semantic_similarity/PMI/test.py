a = [1, 2]
b = [3, 4]
x = len([x for x in a if x in b])
if x == 0:
    print("没有公共元素")

