import os
from typing import List
from google import genai
from dotenv import load_dotenv
import inspect
import wikipedia
import openmeteo_requests   
import sys  


class TextColor: 
    MODEL = '\033[94m'  # Blue
    OBSERVATION = '\033[93m'  # Yellow
    ANSWER = '\033[92m'  # Green
    RESET = '\033[0m'     # Reset color
    TERMINATE = '\033[91m'  # Red
    QUERY = '\033[95m'  # Purple

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
            result, continue_loop = self.loop()
            print(f"{result}\n")
            if not continue_loop:
                break

        return self.message_history()

    def loop(self) -> tuple[str, bool]:
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



if __name__ == "__main__":

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # query = "What is the temperature in fahrenheit right now in Chicago?"
        # query = "What is (3 - 100) * 3 * (50 - 10) + 7?" 
        # query = "What is the weather in London and what is 3 times that temperature plus 3?" 
        # query = "Did more americans die in WW1 or WW2?"
        query = "What is the current tempurature (in fahrenheit) in the city that won the superbowl in 1995?"

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

    agent = TinyReAct(tools=tools, ttl=10)
    result = agent.query(query)
