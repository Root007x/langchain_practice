from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

from dotenv import load_dotenv
load_dotenv()


model_1 = ChatGroq(model = "meta-llama/llama-4-scout-17b-16e-instruct")

llm_endpoint = HuggingFaceEndpoint(
    repo_id= "google/gemma-2-2b-it",
    task =  "text-generation"
)


model_2 = ChatHuggingFace(llm=llm_endpoint)

class Feedback(BaseModel):
    sentiment : Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback")



parser = StrOutputParser()
parser_2 = PydanticOutputParser(pydantic_object=Feedback)

prompt_1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into only positive or negative \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction" : parser_2.get_format_instructions()}
)


classifier_chain = prompt_1 | model_2 | parser_2

# print(classifier_chain.invoke({"feedback" : "This is a wonderful phone"}).sentiment)

prompt_2 = PromptTemplate(
    template="Write an appropriate response based on positive feedback \n {feedback}",
    input_variables=["feedback"]
)

prompt_3 = PromptTemplate(
    template="Write an appropriate response based on negative feedback \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == "positive", prompt_2 | model_2 | parser ), # if
    (lambda x : x.sentiment == "negative", prompt_3 | model_2 | parser), #  elif ...
    RunnableLambda(lambda x : "Could not find sentiment") #  else
)

chain = classifier_chain | branch_chain


print(chain.invoke({"feedback" : "This is a bad phone"}))

print(chain.get_graph().print_ascii())