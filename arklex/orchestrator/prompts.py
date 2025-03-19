RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"

REACT_INSTRUCTION = """
# Instruction
You need to act as an agent that use a set of tools to help the user according to the policy.

# Conversation record
{conversation_record}

# Available tools
{available_tools}

Your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

You current task is:
{task}

Make the decision based on the current task, conversation record, and available tools. If the task has not been finished and available tools are helpful for the task, you should use the appropriate tool to finish it instead of directly give a response.

Thought:
"""