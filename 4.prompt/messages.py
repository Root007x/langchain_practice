from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv


load_dotenv()

llm_endpoint = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V3-0324",
    task =  "text-generation"
)

model = ChatHuggingFace(llm=llm_endpoint)

messages = [
    SystemMessage(content= "Your are a helpful assistant"),
    HumanMessage(content = "Tell me about langchain")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)