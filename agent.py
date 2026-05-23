from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tools import my_tools
import json

class SimpleAgentExecutor:
    def __init__(self, llm, tools, verbose=True):
        self.llm = llm.bind_tools(tools)
        self.tools = {t.name: t for t in tools}
        self.verbose = verbose

    def invoke(self, inputs):
        query = inputs["input"]
        messages = [HumanMessage(content=query)]
        res = self.llm.invoke(messages)

        if res.tool_calls:
            messages.append(res)
            for tool_call in res.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                if self.verbose: print(f"\n[Calling Tool: {name} with {args}]")

                observation = self.tools[name].invoke(args)
                messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

            final_res = self.llm.invoke(messages)
            return {"output": final_res.content}

        return {"output": res.content}

def get_agent():
    # Using gemini-flash-latest based on available model list
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
    return SimpleAgentExecutor(llm=llm, tools=my_tools, verbose=True)
