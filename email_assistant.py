import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.pydantic_v1 import BaseModel as LCBaseModel
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class Parties(BaseModel):
    client: str = Field(description="The name of the client organization")
    counterparty: str = Field(description="The name of the other party in the agreement")

class AgreementReference(BaseModel):
    type: str = Field(description="Type of agreement (e.g., Master Services Agreement)")
    date: str = Field(description="Date of the agreement")

class EmailAnalysisSchema(BaseModel):
    intent: str = Field(description="The primary intent of the email")
    primary_topic: str = Field(description="The main legal topic discussed")
    parties: Parties
    agreement_reference: AgreementReference
    questions: List[str] = Field(description="List of specific questions asked in the email")
    requested_due_date: str = Field(description="The date by which advice is requested")
    urgency_level: str = Field(description="Inferred urgency: low, medium, or high")

class LegalEmailAssistant:
    def __init__(self, api_key: str = None, model_name: str = "gpt-4-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.use_mock = not (self.api_key and LANGCHAIN_AVAILABLE)
        
        if self.use_mock:
            print("Running in MOCK MODE (No API Key found or LangChain missing).")
            print("Returning simulated data for demonstration purposes.\n")
        else:
            self.llm = ChatOpenAI(api_key=self.api_key, model=model_name, temperature=0)

    def analyze_email(self, email_text: str) -> Dict[str, Any]:
        """
        Analyzes the raw email and returns structured JSON output.
        """
        if self.use_mock:
            return self._mock_analysis_result()

        structured_llm = self.llm.with_structured_output(EmailAnalysisSchema)
        
        system_prompt = "You are a legal AI assistant. Extract structured data from the provided email."
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        chain = prompt | structured_llm
        result = chain.invoke({"input": email_text})
        
        return result.dict()

    def draft_reply(self, email_text: str, analysis: Dict[str, Any], contract_text: str) -> str:
        """
        Drafts a legal reply based on analysis and contract text.
        """
        if self.use_mock:
            return self._mock_draft_result(analysis)

        template = """
        You are a professional lawyer. Draft a reply to the following email based on the provided analysis and contract clauses.
        
        CONTEXT:
        - Your Client: {client}
        - Sender Name (address them): Derived from email signature
        - Contract Clauses:
        {contract_text}

        INCOMING EMAIL:
        {email_text}

        ANALYSIS DATA:
        {analysis_json}

        REQUIREMENTS:
        1. Use a professional legal tone.
        2. Clearly answer the specific questions identified in the analysis.
        3. Cite specific clauses (e.g., 9.1, 9.2, 10.2) to support your answers.
        4. Explicitly mention that repeated failure to meet delivery timelines is a material breach.
        5. State the notice period clearly.
        6. Do NOT admit liability.
        7. Keep it concise.
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm

        response = chain.invoke({
            "client": analysis['parties']['client'],
            "contract_text": contract_text,
            "email_text": email_text,
            "analysis_json": json.dumps(analysis, indent=2)
        })

        return response.content

    # --- Mocks for Demonstration without API Key ---
    def _mock_analysis_result(self):
        return {
            "intent": "legal_advice_request",
            "primary_topic": "termination_for_cause",
            "parties": {
                "client": "Acme Technologies Pvt. Ltd.",
                "counterparty": "Brightwave Solutions LLP"
            },
            "agreement_reference": {
                "type": "Master Services Agreement",
                "date": "10 March 2023"
            },
            "questions": [
                "Whether we are contractually entitled to terminate for cause on the basis of repeated delays in delivery",
                "The minimum notice period required"
            ],
            "requested_due_date": "18 November 2025",
            "urgency_level": "high"
        }

    def _mock_draft_result(self, analysis):
        return (
            f"Dear Ms. Sharma,\n\n"
            f"Thank you for your email regarding the {analysis['agreement_reference']['type']} dated "
            f"{analysis['agreement_reference']['date']}.\n\n"
            "Regarding your query on termination for cause: Under Clause 9.2 of the Agreement, "
            "repeated failure to meet delivery timelines explicitly constitutes a 'material breach.' "
            "Therefore, you are contractually entitled to terminate the agreement on these grounds.\n\n"
            "As per Clause 9.1 read in conjunction with Clause 10.2, the minimum notice period required "
            "to effect this termination is thirty (30) days' prior written notice.\n\n"
            "Please let us know if you would like our assistance in drafting the formal notice.\n\n"
            "Regards,\n"
            "Legal Team"
        )

# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Input Data
    SAMPLE_EMAIL = """Subject: Termination of Services under MSA
Dear Counsel,
We refer to the Master Services Agreement dated 10 March 2023 between Acme
Technologies Pvt. Ltd. (“Acme”) and Brightwave Solutions LLP (“Brightwave”).
Due to ongoing performance issues and repeated delays in delivery, we are considering
termination of the Agreement for cause with effect from 1 December 2025.
Please confirm:
1. Whether we are contractually entitled to terminate for cause on the basis of repeated
delays in delivery;
2. The minimum notice period required.
We would appreciate your advice by 18 November 2025.
Regards,
Priya Sharma
Legal Manager, Acme Technologies Pvt. Ltd."""

    CONTRACT_SNIPPET = """Clause 9 – Termination for Cause
9.1 Either Party may terminate this Agreement for cause upon thirty (30) days’ written
notice if the other Party commits a material breach.
9.2 Repeated failure to meet delivery timelines constitutes a material breach.
Clause 10 – Notice
10.1 All notices shall be given in writing and shall be effective upon receipt.
10.2 For termination, minimum thirty (30) days’ prior written notice is required."""

    # 2. Initialize Assistant
    # Note: Pass api_key="sk-..." here if you have one, otherwise it runs Mock mode.
    assistant = LegalEmailAssistant()

    print("--- PART 1: EMAIL ANALYSIS ---")
    analysis_result = assistant.analyze_email(SAMPLE_EMAIL)
    print(json.dumps(analysis_result, indent=2))


    print("\n--- PART 2: DRAFT REPLY ---")
    draft_reply = assistant.draft_reply(SAMPLE_EMAIL, analysis_result, CONTRACT_SNIPPET)
    print(draft_reply)
