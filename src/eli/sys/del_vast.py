import argparse
import os
import time
import requests

"""
Call this script after long running job to delete the vast instance and save money :D
"""

# Set up argument parser
parser = argparse.ArgumentParser(description="Delete Vast.ai instance after optional delay.")
parser.add_argument("--delay", type=float, default=0, help="Delay in hours before deleting the instance")
args = parser.parse_args()

# Wait for the specified delay if provided
if args.delay > 0:
    delay_seconds = args.delay * 3600
    print(f"Waiting for {args.delay} hours before deleting instance...")
    time.sleep(delay_seconds)

vast_instance_id = os.environ["VAST_CONTAINERLABEL"].split("C.")[1]
bearer_token = os.environ["VAST_BEARER_TOKEN"]

payload = {}
url = f"https://console.vast.ai/api/v0/instances/{vast_instance_id}/"
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {bearer_token}",
}

response = requests.request("DELETE", url, headers=headers, data=payload)
