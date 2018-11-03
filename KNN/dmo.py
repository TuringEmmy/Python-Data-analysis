import csv
"""
这是一个测试案例文件
"""
with open('./data.txt','rb') as file:
    print(file.read(1024))

print('-'*100)
with open(r'./data.txt', 'rt') as csvfile:
    # 读取所有的行
    lines = csv.reader(csvfile)
    dataset = list(lines)
    print(dataset)

dic= {'name':'d',"age":18}
dic.items()