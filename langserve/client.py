import requests

response = requests.post("https://20e9-103-101-98-28.ngrok-free.app/chain/invoke",
json={
  "input": {
    "language": "bn",
    "text": "how are you?"
  },
  "config": {},
  "kwargs": {
    "additionalProp1": {}
  }
})

print(response.json()['output'])

