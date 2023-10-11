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
