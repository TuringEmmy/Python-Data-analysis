# -*- coding: utf-8 -*-
import urllib
import re
import MySQLdb

conn = MySQLdb.connect(host = 'mysql.litianqiang.com',port = 7150,user = 'soft',passwd = '123456',db = 'soft',charset = 'utf8',)#连接数据库
cur = conn.cursor()

def getList(page):
    html = urllib.urlopen('http://www.ygdy8.net/html/gndy/dyzz/list_23_%s.html' %page)
    text = html.read()#gbk > gb2312
    text = text.decode('gbk','ignore').encode('utf-8')#解码:把gbk的编码转换为unicode
    reg = r'<a href="(.+?)" class="ulink">(.+?)</a>'
    return re.findall(reg,text)

def getContent(url):
    html = urllib.urlopen('http://www.ygdy8.net%s' %url).read()
    con_text = html.decode('gbk','ignore').encode('utf-8')
    reg = r'<div class="co_content8">(.+?)<p><strong><font color="#ff0000" size="4">'
    reg = re.compile(reg,re.S)#编译正则表达式为对象,增加匹配效率
    text = re.findall(reg,con_text)
    if text:
        text = text[0]
    reg = r'<td style="WORD-WRAP: break-word" bgcolor="#fdfddf"><a href="(.+?)"'
    link = re.findall(reg,con_text)[0]
    return text,link

for i in range(1,159):
    for url,title in getList(page=i):#getList()=列表[(url,标题),(url2,标题2)]
        #i=(url,标题)
        print('正在爬取第%s页的%s' %(i,title))

        content,link = getContent(url)
        print ('正在保存第%s页的%s' %(i,title))
        cur.execute("insert into movie(id,title,content,link) values (NULL ,'%s' ,'%s' ,'%s')" %(title,content.replace("'",r"\'"),link))#执行sql语句
        conn.commit()
        #url,content,link都拿到了