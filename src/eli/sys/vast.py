import os
import requests
import json


vast_instance_id = os.environ["VAST_CONTAINERLABEL"].split("C.")[1]
print(f"vast_instance_id: {vast_instance_id}")
bearer_token = os.environ["VAST_BEARER_TOKEN"]

payload = {}
url = f"https://console.vast.ai/api/v0/instances/{vast_instance_id}/ssh/"
headers = {
   'Accept': 'application/json',
   'Content-Type': 'application/json',
   'Authorization': f'Bearer {bearer_token}'
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)