import os
import pyowm
from dotenv import load_dotenv

from langchain_community.agent_toolkits.amadeus.toolkit import AmadeusToolkit
# AmadeusToolkit.model_rebuild()
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
# from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
import streamlit as st



from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from langchain import hub

load_dotenv()
AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_END_POINT"],
    azure_deployment=os.environ["deployment_name"],
    openai_api_version=os.environ["API_VERSION"],
)

from amadeus import Client
# AmadeusToolkit.model_rebuild()
from langchain_community.tools.amadeus.closest_airport import AmadeusClosestAirport
 
AmadeusClosestAirport.model_rebuild()

from langchain_community.tools.amadeus.flight_search import AmadeusFlightSearch
 
AmadeusFlightSearch.model_rebuild()
from langchain.tools import Tool

amadeus_client = Client(
    client_id = AMADEUS_CLIENT_ID,
    client_secret = AMADEUS_CLIENT_SECRET
)

amadeus_toolkit = AmadeusToolkit(client=amadeus_client,llm=llm)
# AmadeusToolkit.model_rebuild()
amadeus_tools = amadeus_toolkit.get_tools()

weather_api = OpenWeatherMapAPIWrapper()
weather_tool = Tool(
    name = "get_current_weather",
    func = weather_api.run,
    description = "Fetch current weather details for a given city name. Example: 'Weather in Mumbai'"
)

tools = amadeus_tools + [weather_tool]


# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=False)
conversation_chain = ConversationChain(
    llm=llm, 
    memory = st.session_state.memory,
    verbose=True
)

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.render import render_text_description_and_args
from langchain_openai import AzureChatOpenAI

# llm = AzureChatOpenAI(temperature=0)

prompt = hub.pull("hwchase17/react-json")
# prompt  = PromptTemplate.from_template(""" 
#     you are a helpful AI travel assistant.
#     Here is the conversation so far:
#     {chat_history}
#     Now the user is asking: {input}
#     Respond helpfully based on the previous conversation if relevant.
                                       

# """)
agent = create_react_agent(
    llm,
    tools,
    prompt,
    tools_renderer=render_text_description_and_args,
    output_parser=ReActJsonSingleInputOutputParser(),
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    memory=st.session_state.memory
)
# agent_executor.invoke({"input": "What is the name of the airport in Cali, Colombia?"})

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pprint

intent_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a smart intent classifier. Clasify user's message into one of the following:
-book_flight: if they want to book or search flight and origin and destination is also provided
-ask_about_options: if they ask any followup questions about best flight or anything related to flights
-general_query: for any other travel-related questions
Just return the intent label: book_flight, ask_about_options, general_query
User message : {query}
"""

)
intent_chain = LLMChain(llm=llm, prompt = intent_prompt)
flight_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
        You are a helpful travel assistant. Given a user message about booking a flight and checking weather
        ,extract the key information and convert it into the format:
        "Book a flight from {{origin}} to {{destination}} on {{date}} for {{adults}} adults and show weather in {{destination}}."
        User message: "{user_input}"
        Respond only in the required format with correct city names and date. Also convert the date in yyyy-mm-dd format.
        If date is not provided by the user, then just add today's date yourself. If number of passengers is not
        given by the user, assume 1 adult passenger and if nothing is mentioned about weather, just add weather of
        the destination city yourself.
        If nothing is mentioned about origin and destination, then leave the prompt as it is and do
        not do any conversion of that prompt.
    """

)
flight_chain = LLMChain(llm=llm,prompt=flight_prompt)
# user_input = input("Where do you want to fly?")
# user_input = "Book a flight from Mumbai to Goa on 2025-07-15 for 2 adults."


st.title("Agentic Travel Planner")
user_input = st.text_input("Ask me about flight booking and weather...", key="input")
def route_user_input(user_input:str):
    intent = intent_chain.run(query=user_input).strip()
    print(intent)
    if intent == "book_flight":
        result = flight_chain.run(user_input)
        formatted_prompt = result.strip()
        st.session_state.memory.chat_memory.add_user_message(user_input)
        final_response = agent_executor.invoke(
            {
                "input": formatted_prompt
            }
        )
        final_output = final_response.get("output", "")
        st.session_state.memory.chat_memory.add_ai_message(final_output)
        st.write("AI output : ", final_output)
    elif intent == "ask_about_options":
        # print("Answering based on memory...")
        response = conversation_chain.predict(input=user_input)
        # print("Response : \n", response)
        st.write(response)
    elif intent == "general_query":
        response = conversation_chain.predict(input = user_input)
        # print(response)
        st.write(response)
    else:
        # print("Could not understand your query")
        st.write("Sorry, I can only help in travel related queries")

        
route_user_input(user_input)

# if user_input:
#     try:
#         result = chain.run(user_input)
#         formatted_prompt = result.strip()
#         st.session_state.memory.chat_memory.add_user_message(user_input)
#         final_response = agent_executor.invoke(
#             {
#                 "input": formatted_prompt
#             }
#         )
#         final_output = final_response.get("output", "")
#         st.session_state.memory.chat_memory.add_ai_message(final_output)
#         st.write("AI output : ", final_output)
#         # conversation.predict(input=final_output)
#         # st.write("Here is your memory buffer:")
#         # st.write(st.session_state.memory.buffer)
#     except Exception as e:
#         st.error(f"Error : {e}")
    
with st.expander("History"):
    for msg in st.session_state.memory.chat_memory.messages:
        st.markdown(f"{msg.type.capitalize()} : {msg.content}")

if st.button("clear memory"):
    st.session_state.memory.clear()


# if user_input:
#     try:
#         response = agent_executor.invoke({"input":user_input})
#         st.write("AI output : ", response["output"])
#     except Exception as e:
#         st.error(f"Error: {e}")




# print(final_output)
# memory.clear()
# pprint.pprint(memory.buffer)


# memory = ConversationBufferMemory()
# conversation = ConversationChain(
#     llm=llm, 
#     memory = memory,
#     verbose=False
# )

# conversation.predict(input=final_output)

# conversation.predict(input = "What do you know about the weather part")