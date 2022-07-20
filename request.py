import requests
url = 'http://localhost:8080/api'
r = requests.post(url,json={})
print(r.json())