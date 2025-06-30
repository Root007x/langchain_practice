from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()


llm_endpoint = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V3-0324",
    task =  "text-generation"
)

model = ChatHuggingFace(llm=llm_endpoint)


chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain_name}"),
    ("human", "Explain in simple terms, what is {topic}")
    # SystemMessage(content = "You are a helpful {domain_name}"),
    # HumanMessage(content = "Explain in simple terms, what is {topic}")
])


prompt = chat_template.invoke({
    "domain_name" : "Computer Expert",
    "topic" : "Task Scheduler"
})


print(prompt)