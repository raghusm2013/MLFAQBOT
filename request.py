import requests

url = 'http://localhost:5000/api'
r = requests.post(url,json={'usr':'What is RTGS Funds Transfer', 'class':'fundstransfer'})

print(r.json())

