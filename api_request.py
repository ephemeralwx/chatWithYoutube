# -*- coding: utf-8 -*-
"""
test API
"""
import requests
import base64

""" def getSoma():
    import pandas as pd
    url = "https://markets.newyorkfed.org/api/soma/summary.json"
    data = requests.get(url).json()
    df = pd.DataFrame(data["soma"]["summary"])
    df.to_csv("soma_summary.csv") """
    
def test_transcribe():
    SERVICE_URL = 'https://chatwithyoutubevideo.onrender.com/store_transcript'  # change to url of you Cloud Run service
    #SERVICE_URL = 'http://127.0.0.1:80/store_transcript'
    resp = requests.post(SERVICE_URL, json={'video_id':"MYCkEn19XR0"})
    if resp.status_code == 200:
        out_content = resp.json().get('message', '')
        print(out_content)
    else:
        print(resp.text)

def test_ask():
    test_transcribe()
    SERVICE_URL = 'https://chatwithyoutubevideo.onrender.com/ask'  # change to url of you Cloud Run service
    #SERVICE_URL = 'http://127.0.0.1:80/ask'  # change to url of you Cloud Run service


    resp = requests.post(SERVICE_URL, json={'video_id':"MYCkEn19XR0",
                                            'query':'What tools do I need to cut my own hair?'})
    if resp.status_code == 200:
        out_content = resp.json().get('answer','')
        print(out_content)
        print("================================")
        print(resp.text)
    else:
        print("Error occurred!")  
        print(resp.status_code)


#test_transcribe()
test_ask()
