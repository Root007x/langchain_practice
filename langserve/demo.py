from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import uvicorn as uv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from dotenv import load_dotenv
load_dotenv()



model = ChatGroq(model = "meta-llama/llama-4-scout-17b-16e-instruct")

parser = StrOutputParser()


template = ChatPromptTemplate.from_messages([
    ('system', "You are a translation assistant. Translate the following text into {language}.Respond with only the translated sentence. Do not include pronunciation, word meanings, or any explanation."),
    ('user' , "{text}")
])

chain = template | model | parser

# app

app = FastAPI(
    title = "Simple Translation",
    version= "1.0",
    description="API server"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

add_routes(
    app,
    chain,
    path="/chain"
)


if __name__ == "__main__":
    uv.run(app, host = "localhost", port=8080)





