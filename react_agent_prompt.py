from langchain_google_vertexai import VertexAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import tool
import requests
import json
from fuzzywuzzy import fuzz
from langchain.memory import ConversationBufferMemory

memory_type = ConversationBufferMemory(memory_key="chat_history")

@tool
def get_length_characters(words):
    """ find length of the character of given words """
    return len(words)

@tool
def get_covid_death_count(state):
    """ returns covid death count for given state """
    data = requests.get("https://api.rootnet.in/covid19-in/stats/latest")
    for x in json.loads(data.content)["data"]["regional"]:
        if fuzz.ratio(state,x["loc"]) > 70:
            state_name= x["loc"]
            death_count = x["deaths"]
            return f"{state_name}: deaths {death_count}"
@tool
def get_covid_discharged_count(state):
    """ returns covid discharged or cured count for given state """
    data = requests.get("https://api.rootnet.in/covid19-in/stats/latest")
    for x in json.loads(data.content)["data"]["regional"]:
        if fuzz.ratio(state,x["loc"]) > 70:
            state_name= x["loc"]
            discharged_count = x["discharged"]
            return f"{state_name}: discharged {discharged_count}"
        
prompt = hub.pull("hwchase17/react")


if __name__ == "__main__":
    llm = VertexAI(model_name="gemini-1.0-pro-002")

    tools=[get_length_characters,get_covid_death_count,get_covid_discharged_count]

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent,memory=memory_type, tools=tools, verbose=True,handle_parsing_errors=True)

    while True:
        input_message = input("Enter prompt: ")
        agent_executor.invoke(input={"input": {input_message}})
