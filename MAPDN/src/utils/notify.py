import os
import requests
from dotenv import load_dotenv


def notify(msg: str):
    try:
        load_dotenv("src/.env")

        bark_url = os.getenv("BARK_URL")
        if not bark_url:
            print("BARK_URL not found in .env.local, cannot send notification")
            return

        url = f"{bark_url}/{msg}"
        print(url)
        response = requests.get(f"{bark_url}{msg}")
        if response.status_code == 200:
            print("Notification sent successfully")
        else:
            print(f"Notification failed, status code: {response.status_code}")

    except Exception as e:
        print(f"Error sending notification: {str(e)}")
