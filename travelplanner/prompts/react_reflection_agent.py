example_prompt = """
Examples of a travel plan response:
****Example structure****
Task: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?

Day: 1
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 2:
Current City: Charlotte
Transportation: -
Departure Time: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
****Example Structure Ends****
"""

task_prompt = """
This is the current state of the task:
**** Task ****
{task}
**** Task Ends ****
"""

plan_prompt = """
This is the current state of the plan:
**** Plan ****
{plan}
**** Plan Ends ****
"""

suggestion_prompt = """
This is a question suggestion from the development team: 
**** Suggestion ****
{suggestion}
**** Suggestion Ends ****
"""
QUESTION_ANSWERING_INSTRUCTION = (
    """You are an expert in customer service for a travel planning agency. Your work is that, given the task, the conversation history, and the current state of the plan, you must decide whether to continue the plan development or to return a message to the user.

Things to keep in mind:
- Be respectful and mindful.
- If you believe more information should be elicited from the user, do so. 
- When eliciting information from the user take into account that the task and preferences, if any, should be clear but the agency is the one that should recollect the information to generate the travel plan.
- Given a task provided by the user, the agency must recollect the information for the travel plan.
- Assume the user has access to the plan information, so you should only focus on the conversation.
- You can use the development path as many times as needed to further advance the planning process.
- Explicitly answer in the format provided below.

This is the structure of the answer you must return:
- In the case of plan development continuation, you must answer like: [('development', False), ('message','The message to transmit to the user')]
- In the case you want to continue the conversation, you must answer like: [('development', False), ('message','The message to transmit to the user')]

This is the amount of time that the development path has been used since an answer to the user was given: {development_concurrent_time}

"""
    + task_prompt
    + plan_prompt
    + suggestion_prompt
)
TASK_DECODER_INSTRUCTION = (
    """You are an expert in customer service for a travel planning agency. Your work is that, given the conversation history and the current state of the plan, you must generate a comprehensive message that outlines clearly the desires of the client for the travel. Take into account that the plan could still be under development. In the example below, the plan has been developed till breakfast on the third day only.\n"""
    + example_prompt
    + """\nIn the example, the 'task' is the initial requirement of the client, however, this could change as the conversation advances. The message you return is an updated version of this 'task'. If you believe that no change should be made, return the same 'task' message.\n"""
    + task_prompt
    + plan_prompt
)
RESEARCH_INSTRUCTION = (
    """You are an expert researcher of epic travel plans. Based on a task and plan sketch you will generate a question that gives continuity to the plan generation. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section as in the example (i.e., from A to B). Solve this task by reasoning about it (though) and returning a final response for the user based on the reasoning step.\n"""
    + example_prompt
    + """\nThe plan generation must use this structure. In the example above the plan ends with the breakfast on the third day, the question that you generate should aid to decide where to lunch for example. You should only worry about generating a good question to use on the tools. If you believe there is no further question to return, explain why.\n"""
    + task_prompt
    + plan_prompt
)
