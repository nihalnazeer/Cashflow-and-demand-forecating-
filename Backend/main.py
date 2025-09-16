from fastapi import FastAPI
import logging

# EDITED: Import the router objects from their specific locations
from api.analysis import historical
from api.predictive import forecasts, insights
from api import contextchain

### EDITED & CORRECTED ###
# The line below was removed because the services.data_loader module
# is no longer part of the project structure.
# from services.data_loader import load_all_data

# Configure logging
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Sales & Demand Intelligence API")

### EDITED & CORRECTED ###
# The startup event was removed because its only function, load_all_data(),
# was part of the removed services module. The dashboard endpoints will now
# need to fetch data directly on each request.
# @app.on_event("startup")
# async def startup_event():
#     """
#     On server startup, load all necessary data files from the 'results'
#     and 'models' directories into an in-memory cache.
#     """
#     load_all_data()

# Include the modular routers for each part of the API
app.include_router(historical.router)
app.include_router(forecasts.router)
app.include_router(insights.router)
app.include_router(contextchain.router)

@app.get("/", tags=["Health Check"])
async def root():
    """A simple health check endpoint to confirm the API is running."""
    return {"message": "Sales & Demand Intelligence API is running."}