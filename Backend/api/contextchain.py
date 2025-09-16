# api/contextchain.py
from fastapi import APIRouter, BackgroundTasks
from config.settings import settings
from celery import Celery
import json

celery_app = Celery("tasks", broker=settings.redis_url)

router = APIRouter(prefix="/contextchain", tags=["Pipelines"])

@celery_app.task
def run_llm_interpretation_pipeline(data: dict):
    # Simulate pipeline: Load from queue, run LLM, embed, log
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    
    prompt = f"Interpret this sales forecast: {data['predictions']}. {data.get('prompt', '')}"
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    narrative = response.choices[0].message.content
    
    # Embed in Chroma
    from db.chromadb_client import add_interpretation
    add_interpretation(narrative, {"forecast_id": data["prediction_id"]}, data["prediction_id"])
    
    # Log to Mongo
    from db.mongodb_client import log_task_result
    log_task_result("generate_llm_interpretation", "llm_task", "SUCCESS", {"narrative": narrative})

@router.post("/trigger/{pipeline_name}")
async def trigger_pipeline(pipeline_name: str, background_tasks: BackgroundTasks, request_data: dict):
    if pipeline_name == "generate_llm_interpretation":
        background_tasks.add_task(run_llm_interpretation_pipeline.delay, request_data)
    return {"status": "Pipeline triggered asynchronously."}