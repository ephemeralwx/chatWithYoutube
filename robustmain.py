import os
import argparse
import shutil
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "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"
#10/10/23 8:31 P.M. OPEN AI CHATGPT key

parser = argparse.ArgumentParser(description="Query a Youtube video:")
parser.add_argument("-v", "--video-id", type=str, help="The video ID from the Youtube video")

try:
    args = parser.parse_args()
    
    if not args.video_id:
        raise ValueError("Video ID is missing or not provided.")
    
    loader = YoutubeLoader(args.video_id)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    # Check if './data' exists before trying to remove it
    if os.path.exists('./data'):
        shutil.rmtree('./data')
    
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

    green = "\033[0;32m"
    white = "\033[0;39m"

    while True:
        query = input(f"{green}Prompt: ")
        if query == "quit" or query == "q":
            break
        if query == '':
            continue
        
        try:
            response = qa_chain({'query': query})
            print(f"{white}Answer: " + response['result'])
        except Exception as e:
            print(f"Error handling the query: {e}")
    
except ValueError as ve:
    print(f"ValueError: {ve}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
