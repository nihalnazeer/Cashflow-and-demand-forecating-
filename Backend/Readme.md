/Backend/
│
├── 📁 api/
│   ├── __init__.py
│   ├── 📁 analysis/
│   │   ├── __init__.py
│   │   └── historical.py       # Endpoints for historical data (from PostgreSQL)
│   └── 📁 predictive/
│       ├── __init__.py
│       ├── forecasts.py        # Endpoints for FUTURE on-demand forecasts
│       └── insights.py         # The "Why Engine" for interpretability
│
├── 📁 config/
│   └── settings.py           # Your central Pydantic settings
│
├── 📁 core/
│   ├── __init__.py
│   ├── model.py              # The Python class definition for YOUR HALSTM model
│   └── forecasting_service.py # The NEW, CORRECT service for live forecasting
│
├── 📁 db/
│   └── postgres_client.py    # Your existing, perfect PostgreSQL connection manager
│
├── 📁 models/                 # Model artifacts
│   ├── best_ha_lstm.pth
│   ├── target_scaler.pkl
│   └── feature_scaler.pkl
│
├── 📄 main.py                   # The main FastAPI application entry point
└── 📄 pydantic_models.py         # Data contracts for your API


contextchain 

# Initialize a new pipeline with sub-schemas and ChromaDB
contextchain init --file schemas/my_pipeline.json --interactive

# Run a pipeline
contextchain run --pipeline-id my_pipeline

# Run a single task
contextchain run-task --pipeline-id my_pipeline --task_id 1

# Initialize a vector DB collection
contextchain vector init --collection my_collection

# Search in a vector DB collection
contextchain vector search --collection my_collection --query "example query"

# Setup a local LLM
contextchain llm setup --model mistral:7b