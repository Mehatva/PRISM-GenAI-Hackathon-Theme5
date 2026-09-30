import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load env
load_dotenv("/Users/mehatva/Desktop/Projects/Hackathons/Gen AI Hackathon 3.0/.env")

class SubIntent(BaseModel):
    intent_type: str = Field(description="The type of intent. Must match one of the dynamically provided tools, or 'chitchat'.")
    parameters: dict = Field(description="The arguments for the intent.")
    is_contradiction: bool = Field(description="True if this intent contradicts or overrides a previous intent in the same utterance.")

class IntentDecomposition(BaseModel):
    sub_intents: list[SubIntent] = Field(description="List of resolved sub-intents in order of execution.")

if __name__ == "__main__":
    test_utterances = [
        "I need a flight to Chicago... wait, no, make that Denver instead.",
        "Can you search for flights to NYC and also look up the manual for the QN90 TV about HDMI ports?",
    ]
    
    for u in test_utterances:
        print(f"\n--- Utterance: {u} ---")
        result = chain.invoke({"utterance": u})
        for sub in result.sub_intents:
            print(f"- {sub.intent_type}: {sub.parameters} (Contradiction: {sub.is_contradiction})")
