# Tiny ReAct

A lightweight implementation of the [ReAct (Reasoning + Acting)](https://arxiv.org/abs/2210.03629) pattern using Google's Gemini API. Tiny ReAct enables AI agents to break down complex tasks through iterative reasoning and action steps.

## Features

- Implements the ReAct pattern for structured reasoning and action taking
- Integrates with Google's Gemini API for language model capabilities  
- Includes tools for basic math operations, Wikipedia searches, and weather queries
- Configurable time-to-live (TTL) to prevent infinite loops

## Structure

Tiny ReAct works in a simple loop: you ask a question, and it thinks through the problem step by step. It uses Gemini to reason, and if it needs to do something (like look up info or do math), it calls a tool. If it hits an error, it tries again. This keeps going until it finds an answer or runs out of tries. 

<img src="react.png" alt="Agent Reasoning Loop" width="600"/>


## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables:
   - Create a `.env` file in the project root
   - Add your Google Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. Run the agent:

```shell
python3 agent.py "What is the current tempurature (in fahrenheit) in the city that won the superbowl in 1995?"
```

Or use the `TinyReAct` agent library directly:

```python
from agent import TinyReAct

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

query = "What is the current tempurature (in fahrenheit) in the city that won the superbowl in 1995?"

result = agent.query(query)
```

4. Example output

```
Query: What is the current tempurature (in fahrenheit) in the city that won the superbowl in 1995?

Thought1: The user is asking for the current temperature in the city that won the Super Bowl in 1995. My first step is to identify the winner of the Super Bowl in 1995. I will use the `wikipedia_summary` tool to find this information, as it provides a concise summary of historical events. This will allow me to identify the team, and subsequently, their home city.
Action1: wikipedia_summary("Super Bowl 1995 winner", sentences=2)
Observation1: Wikipedia page for 'Super Bowl 1995 winner' not found.

Thought2: My previous attempt to find the Super Bowl 1995 winner failed because the specific page title wasn't found. I need to broaden my search to get the correct information. I will try searching for "Super Bowl 1995" to get a more general summary of the event, which should include the winning team. This approach is more likely to yield results for historical events.
Action2: wikipedia_summary("Super Bowl 1995", sentences=2)
Observation2: Super Bowl XXIX was an American football game between the American Football Conference (AFC) champion San Diego Chargers and the National Football Conference (NFC) champion San Francisco 49ers to decide the National Football League (NFL) champion for the 1994 season. The 49ers defeated the Chargers by a score of 49–26, the most points ever earned in a Super Bowl game, becoming the first team to win five Super Bowl championships.

Thought3: I have successfully identified that the San Francisco 49ers won the Super Bowl in 1995. This means I need to find the temperature in San Francisco. To do this, I first need to get the geographical coordinates (latitude and longitude) of San Francisco. I will use the `wikipedia_coordinates` tool to retrieve these coordinates, as it's specifically designed for location data from Wikipedia.
Action3: wikipedia_coordinates("San Francisco")
Observation3: (Decimal('37.78333332999999782941813464276492595672607421875'), Decimal('-122.4166666699999979073254507966339588165283203125'))

Thought4: I have successfully retrieved the coordinates for San Francisco: latitude 37.78333333 and longitude -122.41666667. Now I can use these coordinates to get the current temperature in San Francisco. I will use the `get_temperature` tool, providing the latitude and longitude as arguments, to retrieve the temperature information.
Action4: get_temperature(37.78333333, -122.41666667)
Observation4: The temperature at 37.78333333, -122.41666667 is 10.290999412536621°C

Thought5: I have successfully retrieved the temperature in San Francisco, which is 10.290999412536621°C. The original query asked for the temperature in Fahrenheit. Therefore, I need to convert this Celsius temperature to Fahrenheit. The formula for converting Celsius to Fahrenheit is (C * 9/5) + 32. I will first multiply the Celsius temperature by 9/5 and then add 32.
Action5: multiply_numbers(10.290999412536621, 1.8)
Observation5: 18.523798942565918

Thought6: I have calculated 10.290999412536621 * 1.8, which is 18.523798942565918. Now, as per the Celsius to Fahrenheit conversion formula (C * 9/5) + 32, I need to add 32 to this result. This will give me the final temperature in Fahrenheit.
Action6: add_numbers(18.523798942565918, 32)
Observation6: 50.52379894256592

Answer: The current temperature in San Francisco, the city that won the Super Bowl in 1995, is approximately 50.52°F.

Here are the steps taken to arrive at this answer:
1. Identified the winner of the Super Bowl in 1995 as the San Francisco 49ers using `wikipedia_summary`.
2. Retrieved the geographical coordinates (latitude and longitude) for San Francisco using `wikipedia_coordinates`.
3. Obtained the current temperature in Celsius for San Francisco using `get_temperature`.
4. Converted the Celsius temperature to Fahrenheit by multiplying by 1.8.
5. Added 32 to the result to get the final temperature in Fahrenheit.
```

## Configuration

- `ttl`: Time-to-live parameter (default: 5) prevents infinite loops by limiting the number of reasoning iterations
- `tools`: List of available tools for the agent to use
- `instruction_prompt`: Customizable prompt template for guiding the agent's behavior

## Available Tools

- `add_numbers(x, y)`: Add two numbers
- `subtract_numbers(x, y)`: Subtract y from x
- `multiply_numbers(x, y)`: Multiply two numbers
- `divide_numbers(x, y)`: Divide x by y
- `get_temperature(lat, lon)`: Get current temperature for coordinates
- `wikipedia_summary(query, sentences)`: Get Wikipedia summary
- `wikipedia_coordinates(page_title)`: Get coordinates for a location
- `search_wikipedia_page(query)`: Search Wikipedia for pages