f=open("data.json","r")

data=f.read()
data=eval(data)
photos=data['photos']
src=[i['src'] for i in photos]
ogsrc=[i['medium'] for i in src]
f.close()

f=open("links.txt","a")
for i in ogsrc:
    print(i,file=f)

f.close()
