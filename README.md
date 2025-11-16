# Legal Email Assistant 

This is a modular Python application designed to analyze incoming legal emails and draft appropriate replies based on specific contract clauses. It uses an Agent-based workflow powered by LLMs (via LangChain).

<h3>Architecture </h3>

The system is divided into two primary modules:

Email Analyzer: Extracts structured metadata (Intent, Parties, Dates, Questions) into JSON format.

Reply Drafter: Generates a professional legal response using the Analysis and specific Contract Clauses.

<h3>Prerequisites </h3>

Python 3.8+

OpenAI API Key (Optional for testing, required for live AI)

<h3>Installation </h3>

Install the required dependencies:

pip install -r requirements.txt


<h3>Usage </h3>

Mock Mode (Default): Simply run the script. If no API key is detected, it uses internal logic to return the expected output for the assignment's sample data. This ensures the code is testable without costs.

Live AI Mode:
Set your OpenAI API key as an environment variable or pass it directly in the code.

export OPENAI_API_KEY="sk-your-api-key-here"
python email_assistant.py

<h3>Dependencies</h3>

langchain-openai: For connecting to GPT models.

pydantic: For enforcing the JSON schema requirements.
