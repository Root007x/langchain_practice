from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv


load_dotenv()


llm_endpoint = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V3-0324",
    task =  "text-generation"
)

model = ChatHuggingFace(llm=llm_endpoint)

chat_history = [
    SystemMessage(content = "You are a helpful AI assistant")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("AI : ",result.content)

print(chat_history)