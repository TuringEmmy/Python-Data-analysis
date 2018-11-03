my_text = 'i have ,been trying !,北京，北京，天安门，五星红旗飘，涮羊肉。'
text=[]
for i in my_text:
    if not i.isspace() :
        text.append(i)
# print(text)
d= {}
for i in text:
    d[i]=text.count(i)
print("关键字\t\t总数")
# print(d)
for key,count  in d.items():
    print(key,'\t\t',count)

