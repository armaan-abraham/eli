import os
import requests
import argparse

"""
Call this script after long running job to delete the vast instance and save money :D
"""

vast_instance_id = os.environ["VAST_CONTAINERLABEL"].split("C.")[1]
bearer_token = os.environ["VAST_BEARER_TOKEN"]

payload = {}
url = f"https://console.vast.ai/api/v0/instances/{vast_instance_id}/"
headers = {
   'Accept': 'application/json',
   'Content-Type': 'application/json',
   'Authorization': f'Bearer {bearer_token}'
}

response = requests.request("DELETE", url, headers=headers, data=payload)

