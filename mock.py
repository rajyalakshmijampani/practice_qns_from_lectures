from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Any, List, Optional
from pydantic import Field

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

faiss_folder = Path("FAISS")

# Function to load FAISS index based on user input week
def load_faiss_for_week(week):
    faiss_index_path = faiss_folder / f"faiss_index_week{week}.pkl"
    if Path(faiss_index_path).exists():
        with open(faiss_index_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"No FAISS index found for Week {week}")
        return None
    
response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question_statement": {"type": "string"},
            "option_a": {"type": "string"},
            "option_b": {"type": "string"},
            "option_c": {"type": "string"},
            "option_d": {"type": "string"},
            "correct_answer": {"type": "string"},
        },
        "required": ["question_statement","option_a","option_b","option_c","option_d","correct_answer"],
    },
}

class GeminiLLM(LLM):

    model_name: str = Field(..., description="gemini-1.5-flash")
    model: Any = Field(None, description="The GenerativeModel instance")

    def __init__(self, model_name):
        super().__init__(model_name=model_name)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name)
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model.generate_content(prompt,
                                               generation_config={"response_mime_type": "application/json",
                                                                  "response_schema": response_schema,
                                                                }
                                                )
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini"

def generate_rag_prompt(query, context, num_questions):
    prompt=("""
    You are Lumi, an AI assistant for the IIT Madras Business Analytics course. You have access to course lectures, transcripts, and all related course materials. 
    Your role is to generate {num_questions} different and random multiple-choice questions (MCQ) based on the given context. 
    Please generate the MCQs in JSON format. Use the `generate_mcqs` function to return the MCQs in JSON format.
    The JSON should be a list where each element is a dictionary representing one MCQ. 
    Each MCQ dictionary must have the following keys:
    - "question_statement": The question text.
    - "option_a": Option A
    - "option_b": Option B
    - "option_c": Option C
    - "option_d": Option D
    - "correct_option": The correct option (A, B, C, or D).

    QUERY: '{query}'
    CONTEXT: '{context}'
    """).format(query=query,context=context,num_questions=num_questions)
    return prompt


class Mock:
        def __init__(self,week):
            self.faiss_vector_store = load_faiss_for_week(week)
            self.gemini_llm =GeminiLLM(model_name='gemini-1.5-flash')
            self.embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        def _warm_up_model(self):
            warm_up_prompt = generate_rag_prompt(query="Hello, how are you?", context="",num_questions=num_questions)
            _ = self.gemini_llm.invoke(warm_up_prompt)

        def generate(self,user_input,num_questions):
            if not self.faiss_vector_store:
                return "No data available for this week."
            
            # Retrieve relevant context
            relevant_docs = self.faiss_vector_store.similarity_search(user_input,k=3)
            context="\n".join([doc.page_content for doc in relevant_docs])

            # Generate prompt and response
            prompt = generate_rag_prompt(query=user_input, context=context, num_questions=num_questions)
            answer=self.gemini_llm.invoke(prompt)
            return answer

week = 1  # User selects Week 1
num_questions = 3
mockbot = Mock(week)
user_input = f"Generate practice questions for week {week}"
print(mockbot.generate(user_input,num_questions))