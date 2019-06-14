import requests
import os

f=open("links.txt","r")
links=f.readlines()
for i in links:
    os.system('wget {}'.format(i))
f.close()