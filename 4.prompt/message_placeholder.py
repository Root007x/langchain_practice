from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# chat template 
chat_template = ChatPromptTemplate([
    ('system', "you are a helpful customer support agent"),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# load chat history
chat_history = []

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())


# print(chat_history)
prompt = chat_template.invoke({
    'chat_history' : chat_history,
    'query' : "where is my refund?"
})

print(prompt)