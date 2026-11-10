import requests

# URL of your running Flask API endpoint
url = "http://127.0.0.1:5000/predict"

# Sample transaction payload matching your dataset's feature columns
sample_transaction = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 181280.00,
    "oldbalanceOrg": 181280.00,
    "newbalanceOrig": 0.00,
    "oldbalanceDest": 0.00,
    "newbalanceDest": 181280.00
}

# Send POST request to Flask
response = requests.post(url, json=sample_transaction)

# Print the JSON response from your AI model
print("API Response Status Code:", response.status_code)
print("API Response JSON:", response.json())
