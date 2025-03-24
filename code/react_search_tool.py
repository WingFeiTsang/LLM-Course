from langchain import hub
from langchain_community.llms import Ollama
from langchain.agents import load_tools
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent

import os
os.environ["SERPAPI_API_KEY"] = "xxxxxx"
# please repace xxxx with your api key

llm = Ollama(model = "deepseek-r1:14b")
tools = load_tools(["serpapi","llm-math"],llm=llm)

template = (
    '尽可能回答以下问题，用中文。如果能力不够，可以使用以下工具,需要时请调用搜索引擎:\n\n'
    '{tools}\n\n '
    'Use the following format:\n\n'
    'Question:the input question you must answer:\n'
    'Thought:you should always think about what to do\n'
    'Action:the action to take, should be one of [{tool_names}]\n'
    'Action Input:the input to the action'
    'Observation:the result of the action'
    '...(this Thought/Action/Action Input/Observation can repeat N times)\n\n'
    'Thought:I now know the final answer\n'
    'Final Answer:the final answer to the original input question'
    'Begin!\n\n'
    'Question:{input}\n'
    'Thought:{agent_scratchpad}'
)

prompt = PromptTemplate.from_template(template)
print(prompt)

agent = create_react_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)
agent_executor.invoke({"input":
                           """目前市场上玫瑰花的一般进货价格是多少？ \n
                           如果我在此基础上加价百分之5，应该如何定价？"""
                       })
print(agent_executor.invoke())