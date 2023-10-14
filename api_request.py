# -*- coding: utf-8 -*-
"""
test API
"""
import requests
import base64


def test_chatyoutube():
    #test_transcribe()
    SERVICE_URL = 'https://chatwithyoutubevideo.onrender.com/chatyoutube'  # change to url of you Cloud Run service
    #SERVICE_URL = 'http://127.0.0.1:5000/chatyoutube'  # change to url of you Cloud Run service
    youtube_url = 'https://www.youtube.com/watch?v=MYCkEn19XR0'

    resp = requests.post(SERVICE_URL, json={'user_id':"MYCkEn19XR0",'youtube_url':youtube_url,
                                            'query':'What tools do I need to cut my own hair?'})
    if resp.status_code == 200:
        #out_content = resp.json().get('answer','')
        #print(out_content)
        #print("================================")
        print(resp.text)
    else:
        print("Error occurred!")  
        print(resp.status_code)

def test_health():
    
    #SERVICE_URL = 'https://chatwithyoutubevideo.onrender.com/health'  # change to url of you Cloud Run service
    SERVICE_URL = 'http://127.0.0.1:80/ask'  # change to url of you Cloud Run service


    resp = requests.get(SERVICE_URL)
    if resp.status_code == 200:
        print(resp.text)
    else:
        print("Error occurred!")  
        print(resp.status_code)
#test_transcribe()
#test_health()
test_chatyoutube()
