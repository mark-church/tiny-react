import os
from typing import Dict, List, Any, Callable
from google import genai
from google.genai import types
from dotenv import load_dotenv
import inspect
import wikipedia
import pdb
import openmeteo_requests   


# Improvements:
# - Try to optimize the prompt for the minimum number of iterations
# - Add similar tools like Google search to see if it improves or worsens the results
# - Create reasoning branches to see if multiple separate reasoning processes give a better result
# - Add a tool to check if the answer is correct
# - Add evaluation to measure optimizations that produce better answers in less steps

class TextColor: 
    MODEL = '\033[94m'  # Blue
    OBSERVATION = '\033[93m'  # Yellow
    ANSWER = '\033[92m'  # Green
    RESET = '\033[0m'     # Reset color
    TERMINATE = '\033[91m'  # Red
    QUERY = '\033[95m'  # Purple

## The Instruction Prompt

The instruction prompt is like the agent's personality and rulebook rolled into one. It tells the agent:
1. How to structure its thinking (using the ReAct pattern)
2. What tools it has available
3. What rules it must follow
4. How to format its responses

The prompt is crucial because it guides the model's behavior and ensures it follows the ReAct pattern correctly. We dynamically insert the available tools into the prompt, so the agent knows exactly what it can do.

instruction_prompt_template = """
Use the following thinking trace pattern and tools to solve the problem in a number of steps.

# TOOLS
{tool_prompt}

# EXAMPLE THINKING TRACE
## INPUT
Query: What is 10 * 3 + 10?

Thought1: First, I need to multiply 10 by 3.
Action1: multiply_numbers(10, 3)

Observation1: 30

## OUTPUT
Thought2: Now I need to add 10 to this result.
Action2: add_numbers(30, 10)

# RULES
1. Do not make up observations. Observations should only be provided as input, never as output.
2. One action per turn. Do not make multiple tool calls or skip steps. Each response should only contain a single Thought, Action, or Answer.
3. Think out loud and feel free to be expressive and explain your thought process in each Thought. Each Thought should be a minimum of 3 sentences.
4. Denote the iteration of each step (as-in Thought1, Action1)
5. When you have the final answer, and ONLY then, use the format:
Answer: [your final answer]
6. When you have the final answer, rephrase the answer in the context of the original query, and briefly summarize the work you did, listing them as numbered steps.
7. If you cannot get the correct information from a tool, try different variations of the query. After 5 attempts, return an error.
8. Stick to the format as shown.
"""

# Define available tools (TOOLS definition remains the same)
def add_numbers(x: int, y: int) -> int:
    """
    Adds two numbers.
    parameter x: the first number to add.
    parameter y: the second number to add.
    returns: the sum of the two numbers.
    example: add_numbers(4, 7)"""
    return x + y

def divide_numbers(x: float, y: float) -> float:
    """
    Divides two numbers.
    parameter x: the first number to divide.
    parameter y: the second number to divide.
    returns: the quotient of the two numbers.
    example: divide_numbers(10, 3)"""
    return x / y

def subtract_numbers(x: int, y: int) -> int:
    """
    Subtract second number from first number.
    parameter x: the first number.
    parameter y: the second number.
    returns: the difference between the two numbers.
    example: subtract_numbers(4, 7)"""
    return x - y

def multiply_numbers(x: int, y: int) -> int:
    """
    Multiply two numbers.
    parameter x: the first number.
    parameter y: the second number.
    returns: the product of the two numbers.
    example: multiply_numbers(4, 7)"""
    return x * y

def get_temperature(latitude: float, longitude: float) -> str:
    """
    Get the temperature for a location.
    parameter latitude: the latitude of the location, represented as a positive number for N latitudes and a negative number for S latitudes
    parameter longitude: the longitude of the location, represented as a positive number for E longitudes and a negative number for W longitudes
    returns: a string containing the temperature
    example: get_temperature(13.1, -97.4)"""

    om = openmeteo_requests.Client()
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["temperature_2m"]
    }

    responses = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    response = responses[0]
    temperature = response.Current().Variables(0).Value()

    return f"The temperature at {latitude}, {longitude} is {temperature}°C"  

def search_wikipedia_page(query: str) -> List[str]:
    """
    Searches Wikipedia for a given query and returns a list of relevant page titles. This is useful for wikipedia_page_content() and wikipedia_coordinates() which require a page_title as input. Try using this usful when frequently getting "Wikipedia page for '{query}' not found" errors.
    parameter query: the search term or question for Wikipedia.
    returns: a list of relevant page titles.
    example: search_wikipedia_page("WW2 casualties")
    """
    try:
        wikipedia.set_user_agent("MyReActAgent/1.0 (myemail@example.com)")
        search_results = wikipedia.search(query, suggestion=True)[0]
        return search_results
    except wikipedia.exceptions.PageError:
        return f"Wikipedia page for '{query}' not found."
    

def wikipedia_coordinates(page_title: str) -> List[str]:
    """
    Provides the coordinates for Wikipedia pages if they are a location. First use search_wikipedia_page to get the page_title.
    parameter page_title: the title of the Wikipedia page to get the coordinates of.
    returns: a latitude and longitude.
    example: wikipedia_coordinates("Tallahassee, FL")
    """
    try:
        wikipedia.set_user_agent("MyReActAgent/1.0 (myemail@example.com)")
        coordinates = wikipedia.page(page_title, auto_suggest=False, redirect=True).coordinates
        return coordinates
    except wikipedia.exceptions.PageError:
        return f"Wikipedia page for '{page_title}' not found."
        

def wikipedia_summary(query: str, sentences: int = 10) -> str:
    """
    Searches Wikipedia for a given query and returns a summary. Returns up to 10 sentences max.
    parameter query: the search term or question for Wikipedia.
    parameter sentences: the number of sentences to try and retrieve for the summary.
    returns: a string containing the summary of the Wikipedia page, or an error message if the page is not found or an error occurs.
    example: wikipedia_summary("United States president", 5)
    """
    try:
        wikipedia.set_user_agent("MyReActAgent/1.0 (myemail@example.com)")
        summary = wikipedia.summary(query, sentences=sentences, auto_suggest=False, redirect=True)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        options = e.options[:3] # Get first 3 options
        return f"The query '{query}' is ambiguous. Did you mean: {', '.join(options)}? Or try a more specific query."
    except wikipedia.exceptions.PageError:
        return f"Wikipedia page for '{query}' not found."
    except Exception as e:
        return f"An error occurred while searching Wikipedia for '{query}': {str(e)}"


class TinyReAct:
    def __init__(self, ttl=5, tools=None):
        self.messages = []
        self.iteration = 0
        self.ttl = ttl
        self.tools = tools

        # Initialize Gemini client
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

        # Formats the instruction prompt by dynamically generating the list of tools
        tool_prompt = ""
        for function_name in self.tools:
            tool_prompt += f"\n\n{function_name.__name__}{inspect.signature(function_name)}{function_name.__doc__}"
        
        self.instruction_prompt = instruction_prompt_template.format(tool_prompt=tool_prompt)


    def query(self, query: str) -> str:
        # Adds the first message to the model with the instruction prompt and user query
        self.messages = [{
            "role": "user",
            "parts": [{"text": self.instruction_prompt + "\n\n#THINKING TRACE\n\nQuery: " + query + "\n"}]
        }]
        
        print(f"{TextColor.QUERY}Query: {query}\n{TextColor.RESET}")

        while True:
            result, continue_loop = self.iterate()
            print(f"{result}\n")
            if not continue_loop:
                break

        return self.message_history()

    def iterate(self) -> tuple[str, bool]:
        # Terminates the overall loop if TTL is reached
        if self.iteration >= self.ttl:
            return f"{TextColor.TERMINATE}Terminate: Reached maximum number of iterations{TextColor.RESET}", False
        
        # Generates the model response
        reasoning = self.reason()
        
        # Check if the model response is empty
        if not reasoning:  
            return "Error: No response from model", False
        # Check if final answer is found
        elif "Answer" in reasoning:
            return f"{TextColor.ANSWER}{reasoning}{TextColor.RESET}", False
        # Check if an action is specified
        elif "Action" in reasoning:
            self.iteration += 1
            action_line = [line.strip() for line in reasoning.split('\n') if line.startswith("Action")][0]
            function_call = action_line.split(":")[1].strip()
            # Executes the action
            observation = self.act(function_call)
            return f"{TextColor.MODEL}{reasoning}{TextColor.RESET}\n{TextColor.OBSERVATION}{observation}{TextColor.RESET}", True
        else:
            return "Error: Answer or Action not found in model response", True

    def reason(self) -> str:
        try:
            # Generates the model response
            response = self.client.models.generate_content(
                contents=self.messages, 
                model="gemini-2.5-flash-preview-05-20"
            )
            # Adds the model response to the conversation history
            self.messages.append({
                "role": "model",
                "parts": [{"text": response.text}]
            })

            return response.text
        except Exception as e:
            error_message = f"Error calling Gemini API: {str(e)}"            
            return error_message

    def act(self, function_call: str) -> str:
        # Extract function call from the model response
        function_name = function_call.split('(')[0].strip()
        
        # Check if function exists in tool list
        if function_name not in [func.__name__ for func in self.tools]:
            return f"\nError: Function '{function_name}' is not in the list of available tools"
        try:
            # Executes the function
            result = eval(f"{function_call}")
            observation = f"Observation{self.iteration}: {result}"
        except Exception as e:
            observation = f"\nError: Cannot execute function {function_call}: {str(e)}"

        # Adds the observation to the conversation history
        self.messages.append({
            "role": "user",
            "parts": [{"text": observation}]
        })

        return observation
    
    def message_history(self) -> str:
        """Return the conversation history as a nicely formatted string."""
        output = []
        output.append("\nConversation History:")
        output.append("-" * 50)
        
        for msg in self.messages:
            if msg["role"] == "user":
                output.append(f"{TextColor.OBSERVATION}{msg['parts'][0]['text']}{TextColor.RESET}")
            elif msg["role"] == "model":
                output.append(f"{TextColor.MODEL}{msg['parts'][0]['text'].strip()}{TextColor.RESET}")
        
        output.append("-" * 50)
        return "\n".join(output)

    
def main():
    tools = [
        add_numbers,
        subtract_numbers,
        multiply_numbers,
        divide_numbers,
        get_temperature,
        search_wikipedia_page,
        wikipedia_coordinates,
        wikipedia_summary
    ]

    # query = "What is the temperature in fahrenheit right now in Chicago?"
    # query = "What is (3 - 100) * 3 * (50 - 10) + 7?" 
    # query = "What is the weather in London and what is 3 times that temperature plus 3?" 
    # query = "Did more americans die in WW1 or WW2?"
    query = "What is the current tempurature (in fahrenheit) in the city that won the superbowl in 1995?"

    agent = TinyReAct(tools=tools, ttl=10)
    result = agent.query(query)


if __name__ == "__main__":
    main()

    
    # # Generate and save the diagram
    # diagram = generate_mermaid_diagram(messages, question, answer_content if answer_content else None)
    # with open("mermaid.md", "w") as f:
    #     f.write("```mermaid\n")
    #     f.write(diagram)
    #     f.write("\n```")
    # print("\nDiagram saved to mermaid.md")



# def generate_mermaid_diagram(messages: List[Dict[str, Any]], question: str, answer: str) -> str:
#     """Generates a mermaid diagram from the messages. The question node, thought node, action node, and observation are different colors.
#        The text of the message is the label of the node. No other text is allowed in the diagram.
#     """
#     def escape_text(text: str) -> str:
#         """Escape special characters for Mermaid nodes."""
#         import re
#         # Replace quotes with single quotes
#         text = text.replace('"', "'")
#         # Replace newlines with spaces
#         text = text.replace('\n', ' ')
#         # Remove any remaining special characters that might cause issues
#         text = text.replace('`', '')
#         text = text.replace('*', '')
#         # Remove the prefix and number completely
#         text = re.sub(r'(Thought|Action|Observation)\d+:\s*', '', text)
#         return text

#     def get_iteration(text: str) -> int:
#         """Extract iteration number from message text."""
#         import re
#         match = re.search(r'(Thought|Action|Observation)(\d+):', text)
#         return int(match.group(2)) if match else 0

#     diagram = ["graph TD"]
#     node_count = 0
#     nodes = []  # Store nodes to avoid duplication
#     iteration_nodes = {}  # Store nodes by iteration
#     final_answer = None
    
#     # Define node styles with a more polished color scheme and smaller font
#     styles = {
#         "question": "fill:#2C3E50,stroke:#34495E,stroke-width:2px,color:#ECF0F1,font-size:12px",  # Dark blue with light text
#         "thought": "fill:#3498DB,stroke:#2980B9,stroke-width:2px,color:#ECF0F1,font-size:12px",    # Blue with light text
#         "action": "fill:#27AE60,stroke:#219A52,stroke-width:2px,color:#ECF0F1,font-size:12px",     # Green with light text
#         "observation": "fill:#E67E22,stroke:#D35400,stroke-width:2px,color:#ECF0F1,font-size:12px"  # Orange with light text
#     }
    
#     # Add question as first node
#     nodes.append((f"node{node_count}", f"Question: {escape_text(question)}", "question"))
#     node_count += 1
    
#     # Process messages, skipping the first one (system prompt + question)
#     for msg in messages[1:]:
#         content = msg["parts"][0]["text"]
        
#         # Split content into lines to handle multi-line messages
#         lines = content.split('\n')
#         for line in lines:
#             # Determine node type and iteration
#             if line.startswith("Thought"):
#                 iteration = get_iteration(line)
#                 if iteration not in iteration_nodes:
#                     iteration_nodes[iteration] = []
#                 iteration_nodes[iteration].append((f"node{node_count}", escape_text(line), "thought"))
#                 node_count += 1
#             elif line.startswith("Action"):
#                 iteration = get_iteration(line)
#                 if iteration not in iteration_nodes:
#                     iteration_nodes[iteration] = []
#                 iteration_nodes[iteration].append((f"node{node_count}", escape_text(line), "action"))
#                 node_count += 1
#             elif line.startswith("Observation"):
#                 iteration = get_iteration(line)
#                 if iteration not in iteration_nodes:
#                     iteration_nodes[iteration] = []
#                 iteration_nodes[iteration].append((f"node{node_count}", escape_text(line), "observation"))
#                 node_count += 1

#     if answer:
#         final_answer = (f"node{node_count}", escape_text(answer), "question")
#         node_count += 1
    
#     # Add question node
#     for node_id, text, style in nodes:
#         diagram.append(f"{node_id}([\"{text}\"]):::{style}")
    
#     # Add iteration subgraphs with dotted borders
#     for iteration, iter_nodes in iteration_nodes.items():
#         diagram.append(f"subgraph Iteration_{iteration}[\"Iteration {iteration}\"]")
#         for node_id, text, style in iter_nodes:
#             diagram.append(f"{node_id}([\"{text}\"]):::{style}")
#         diagram.append("end")
    
#     # Add final answer node if it exists
#     if answer:
#         node_id, text, style = final_answer
#         diagram.append(f"{node_id}([\"{text}\"]):::{style}")
    
#     # Add connections between nodes
#     all_nodes = nodes + [node for nodes in iteration_nodes.values() for node in nodes]
#     if answer:
#         all_nodes.append(final_answer)
#     for i in range(len(all_nodes) - 1):
#         diagram.append(f"node{i} --> node{i+1}")
    
#     # Add style definitions
#     for style_name, style_def in styles.items():
#         diagram.append(f"classDef {style_name} {style_def}")
#     # Add subgraph style
    
#     diagram.append("classDef iteration stroke-dasharray: 5 5,stroke:#666,stroke-width:2px,fill:none,text-align:right")
    
#     return "\n".join(diagram)


# instruction_prompt_template2 = """You are an AI assistant that strictly follows the ReAct (Reasoning and Acting) pattern.
# You will solve problems by breaking them down into a sequence of Thought, Action, PAUSE, and Observation steps.

# **Instructions:**
# 1.  **Thought**: Reason about the problem and what single step is needed next.
# 2.  **Action**: Specify ONE function call to perform that step. Format: `Action: function_name(arg1, arg2)`
# 3.  **PAUSE**: You MUST output `PAUSE` on a new line immediately after the Action line. You will then STOP and wait for an Observation.

# I will provide an `Observation: [result of your action]` after your PAUSE.
# Then, you will continue the cycle:
# 1.  **Thought_iteration**: Reason about the Observation and decide the next step.
# 2.  **Action_iteration**: (If needed) `Action: function_name(arg1, arg2)`
# 3.  **PAUSE**:

# Repeat this process until you have the final answer.

# When you have the final answer, and ONLY then, use the format:
# Answer: [your final answer]

# **Available tools:**
# {tool_prompt}

# **Crucial Rules:**
# -   **One Action Per Turn**: You MUST output only ONE `Action: ...` line, followed immediately by `PAUSE` on the next line. Do NOT output multiple actions or the final answer in the same turn as an action.
# -   **Your response should only contain a single Thought, Action, and/or PAUSE.**
# -   **Wait for Observation**: After `PAUSE`, you MUST wait for the `Observation:` from me. Do not invent results.
# -   **Format Adherence**: Stick to the exact formats shown.
# -   **Denote Iterations**: Show the iteration of each step (as-in Thought1)

# **Example showing progressive steps:**

# Query: What is 3 times the temperature in London plus 10?

# # STEP_1
# Thought1: First, I need to get the weather in London.
# Action1: get_weather("London")
# PAUSE

# Observation1: In London it is sunny and 11°C

# # STEP_2
# Question: What is 3 times the temperature in London plus 10?

# Thought1: First, I need to get the weather in London.
# Action1: get_weather("London")
# PAUSE

# Observation1: In London it is sunny and 11°C

# Thought2: Now I need to multiply the temperature of 11 by 3.
# Action2: multiply_numbers(11, 3)
# PAUSE

# Observation2: 33

# # STEP_3
# Question: What is 3 times the temperature in London plus 10?

# Thought1: First, I need to get the weather in London.
# Action1: get_weather("London")
# PAUSE

# Observation1: In London it is sunny and 11°C

# Thought2: Now I need to multiply the temperature of 11 by 3.
# Action2: multiply_numbers(11, 3)
# PAUSE

# Observation2: 33

# Thought3: Finally, I need to add 10 to this result.
# Action3: add_numbers(33, 10)
# PAUSE

# Observation3: 43

# Thought4: I have the final result.
# Answer: The temperature in London is 11°C. Three times that is 33°C, and adding 10 gives us 43°C.

# **Remember to:**
# - Always think before acting.
# - Use exact function names and argument formats.
# - Include units in final answers if appropriate.
# - Only perform one action at a time, then PAUSE and wait. There should NEVER be multiple PAUSEs in your response.
# - Your response should contain, at most, one Thought, Action, and/or PAUSE.
# """
