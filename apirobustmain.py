'''
from flask import Flask, request, jsonify
import os
import re
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi

app = Flask(__name__)

# Setting up OpenAI API key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# We'll use a dictionary to store video content in-memory
video_data = {}

@app.route('/ask', methods=['POST'])
def ask():
    video_link = request.json.get('video_link')
    query = request.json.get('query')

    # Extract video_id from the link
    video_id_match = re.search(r'v=([^&]+)', video_link)
    if video_id_match:
        video_id = video_id_match.group(1)
    else:
        return jsonify({"error": "Video ID not found in the URL."}), 400

    if not video_id or not query:
        return jsonify({"error": "video_id or query missing"}), 400

    # If the video's content is not in memory, fetch and process it
    if video_id not in video_data:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text_values = [item['text'] for item in transcript]
        documents = ' '.join(text_values)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents([documents])

        vectordb = Chroma.from_documents(
            documents,
            embedding=OpenAIEmbeddings(),
            persist_directory='./data'
        )
        vectordb.persist()

        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            verbose=False
        )

        # Store in memory
        video_data[video_id] = qa_chain

    else:
        qa_chain = video_data[video_id]

    try:
        response = qa_chain({'query': query})
        return jsonify({"answer": response['result']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
'''
from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

import torch

app = Flask(__name__)

# Load a fine-tuned BERT model and tokenizer
MODEL_NAME = "distilbert-base-uncased-distilled-squad"

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForQuestionAnswering.from_pretrained(MODEL_NAME)


# Optimize model for inference
model.eval()

# In-memory storage for video transcripts (in a real-world scenario, consider using databases like PostgreSQL, MongoDB, etc.)
video_data = {}

@app.route('/store_transcript', methods=['POST'])
def store_transcript():
    video_id = request.json.get('video_id')
    transcript = request.json.get('transcript')

    if not video_id or not transcript:
        return jsonify({"error": "video_id or transcript missing"}), 400
    
    video_data[video_id] = transcript
    return jsonify({"message": "Transcript stored successfully!"})

@app.route('/ask', methods=['POST'])
def ask():
    video_id = request.json.get('video_id')
    query = request.json.get('query')

    if not video_id or not query:
        return jsonify({"error": "video_id or query missing"}), 400

    # Retrieve the video transcript
    transcript = video_data.get(video_id)
    if not transcript:
        return jsonify({"error": "No transcript found for the given video_id"}), 400

    # Tokenize and predict
    with torch.no_grad():
        inputs = tokenizer.encode_plus(query, transcript, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"].tolist()[0]
        output = model(**inputs)
        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits)

    # Extract the answer
    answer = tokenizer.decode(input_ids[answer_start:answer_end+1], skip_special_tokens=True)

    if answer:
        return jsonify({"answer": answer})
    else:
        return jsonify({"error": "Unable to extract a clear answer"}), 400

if __name__ == '__main__':
    app.run()
