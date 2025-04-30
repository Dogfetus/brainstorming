import requests
import uuid
import json

# the uuid is unique to each device (so each user needs their own uuid to not logut other users)
# we might not need to save the uuid (since we dont know which user is trying to login, so we dont know which uuid to retreive)
# to login:
new_uuid = str(uuid.uuid4())
print("Generated UUID:", new_uuid)

url = "https://data.stepmaniax.com/sign/in"

headers = {
    "accept": "*/*",
    "content-type": "application/json",
    "accept-encoding": "gzip, deflate, br",
    "user-agent": "StepManiaX/1 CFNetwork/1568.200.51 Darwin/24.1.0",
    "accept-language": "en-GB,en;q=0.9"
}

data = {
    "account": "nsfw",
    "password": "KaLkUn2003",
    "uuid": new_uuid,
    "apiVersion": 6
}

response = requests.post(url, json=data, headers=headers)
json_response = response.json()

print("Status Code:", response.status_code)
print("Response Body:", json.dumps(json_response, indent=2))
print(json_response["auth_token"])
print(json_response["account"]["id"])






# to logout:
# data from the login response:
gamer_id = json_response["account"]["id"]
auth_token = json_response["auth_token"]


signout_url = "https://data.stepmaniax.com/sign/out"

headers = {
    "accept": "*/*",
    "content-type": "application/json",
    "accept-encoding": "gzip, deflate, br",
    "user-agent": "StepManiaX/1 CFNetwork/1568.200.51 Darwin/24.1.0",
    "accept-language": "en-GB,en;q=0.9"
}

signout_data = {
    "apiVersion": 6,
    "auth_gamer": gamer_id,
    "auth_token": auth_token
}

response = requests.post(signout_url, json=signout_data, headers=headers)
json_response = response.json()
