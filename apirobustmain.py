from flask import Flask, request, jsonify
import os
import shutil
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

app = Flask(__name__)

# We'll use a dictionary to store video content in-memory
video_data = {}

@app.route('/ask', methods=['POST'])
def ask():
    video_id = request.json.get('video_id')
    query = request.json.get('query')

    if not video_id or not query:
        return jsonify({"error": "video_id or query missing"}), 400

    # If the video's content is not in memory, fetch and process it
    if video_id not in video_data:
        loader = YoutubeLoader(video_id)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(documents)

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
