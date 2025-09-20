from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware # ✨ 1. Import the CORS middleware
import logging

# Your existing router and DB imports
from api.analysis import historical
from api.predictive import forecasts, insights
from api import contextchain
from db.postgres_client import init_pool, close_pool

# Configure logging
logging.basicConfig(level=logging.INFO)

# ✨ 2. Define the 'lifespan' function for modern startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Initializes the database pool on startup and closes it on shutdown.
    """
    logging.info("Application startup...")
    await init_pool()
    yield
    logging.info("Application shutdown...")
    await close_pool()

# Create the FastAPI app instance, passing in the new lifespan manager
app = FastAPI(
    title="Sales & Demand Intelligence API",
    lifespan=lifespan
)


# ✨ 3. Add the CORS Middleware to the application
# Define the list of origins that are allowed to make requests to this API
origins = [
    "http://localhost:3000",  # The default address for a Next.js frontend
    # You can add your deployed frontend URL here later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # Allow specific origins
    allow_credentials=True,       # Allow cookies to be included in requests
    allow_methods=["*"],          # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],          # Allow all headers
)


# Include your existing routers (no changes here)
app.include_router(historical.router)
app.include_router(forecasts.router)
app.include_router(insights.router)
app.include_router(contextchain.router)


@app.get("/", tags=["Health Check"])
async def root():
    """A simple health check endpoint to confirm the API is running."""
    return {"message": "Sales & Demand Intelligence API is running."}

# The old @app.on_event decorators are no longer needed
# as their logic has been moved into the 'lifespan' function.