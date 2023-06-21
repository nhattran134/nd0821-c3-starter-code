import json
import requests

response = requests.get('http://udacity-nd0821-prod.eba-misicvr9.us-east-1.elasticbeanstalk.com/', data=json.dumps(data))

print(response.status_code)
print(response.json())