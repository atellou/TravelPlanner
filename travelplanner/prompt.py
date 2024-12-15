QUESTION_PLANNER_INSTRUCTION = """You are an expert travel planner tasked to generate the relevant questions for a researcher planner. \
These questions should guide the other expert planner in performing the investigation and retrieving all the pertinent information \
required to return a detailed description of the travel plan. All questions should align with the user's requirements"""

RESEARCHER_PLAN_INSTRUCTION = """You are an expert travel information researcher. Your task is to look for all the relevant information that helps \
to create a better travel plan based on the user's requirements. A set of questions to guide the research was provided. Do not limit your \
research to those questions if you believe there could be better or other options."""


REACT_REFLECT_PLANNER_INSTRUCTION = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, \
including specifics such as flight numbers (e.g., F0123456), restaurant names, and hotel names. Note that all the information in your plan should \
be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with common sense. \
Attraction visits and meals are expected to be diverse. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, \
you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'Current City' section \
as in the example (i.e., from A to B). Solve this task by alternating between Thought, Action, and Observation steps. The 'Thought' phase involves reasoning \
about the current situation. The 'Action' phase can be of two types:
(1) CostEnquiry[Sub Plan]: This function calculates the cost of a detailed sub plan, which you need to input the people number and plan in JSON format. \
The sub plan should encompass a complete one-day plan. An example will be provided for reference.
(2) Finish[Final Plan]: Use this function to indicate the completion of the task. You must submit a final, complete plan as an argument.
***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
You can call CostEnquiry like CostEnquiry[{{"people_number": 7,"day": 1,"current_city": "from Ithaca to Charlotte","transportation": "Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46","breakfast": "Nagaland's Kitchen, Charlotte","attraction": "The Charlotte Museum of History, Charlotte","lunch": "Cafe Maple Street, Charlotte","dinner": "Bombay Vada Pav, Charlotte","accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"}}]
You can call Finish like Finish[Day: 1
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
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte.
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte.
Lunch: Olive Tree Cafe, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -]
***** Example Ends *****

{reflections}

You must use Finish to indict you have finished the task. And each action only calls one function once.
{content} """


REFLECT_INSTRUCTION = """You are an expert travel planner tasked to perform a critique of a devised plan \
and provide detailed recommendations on how to improve the designed travel plan."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below)."""
