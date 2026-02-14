from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.processor import preprocess_text
from src.model_logic import query_database

app = FastAPI(
    title="Troll API",
    description="API for analyzing comments and finding similar entries in a vector database.",
    version="1.0.0"
)

class AnalysisRequest(BaseModel):
    comment: str

@app.get("/")
async def root():
    return {"status": "online", "model": "all-MiniLM-L6-v2"}

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    try:
        cleaned_text = preprocess_text(request.comment)
        search_results = query_database(cleaned_text)
        
        return {
            "query": request.comment,
            "processed_query": cleaned_text,
            "matches": search_results['documents'][0],
            "distances": search_results['distances'][0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))