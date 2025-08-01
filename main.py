"""FastAPI application for Document AI Agent with Ollama integration."""

import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiofiles

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from core.ai_agent import DocumentAIAgent
from core.ollama_llm import validate_ollama_setup, model_manager
from config import config, update_config_from_payload

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.get("app.log_level", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config.get("app.name", "Document AI Agent"),
    description=config.get("app.description", "Intelligent document processing with Ollama & LangChain"),
    version=config.get("app.version", "1.0.0")
)

# Initialize AI agent
ai_agent = DocumentAIAgent()

# Templates
templates = Jinja2Templates(directory=str(config.get_paths()["templates"]))

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str = Field(..., description="Question to ask about the documents")
    options: Optional[Dict[str, Any]] = Field(default={}, description="Additional options")

class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any] = Field(..., description="Configuration update payload")

class ModelRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to manage")
    action: str = Field(..., description="Action to perform: pull, delete, info")

class QuestionResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    error: Optional[str] = None
    model_used: Optional[str] = None
    processing_time: Optional[float] = None

class StatusResponse(BaseModel):
    status: str
    ollama_status: Dict[str, Any]
    agent_stats: Dict[str, Any]
    config_summary: Dict[str, Any]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config": config._config
    })

@app.post("/upload", response_class=JSONResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process multiple documents."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        uploaded_files = []
        failed_files = []

        max_file_size = config.get("document_processing.max_file_size", 52428800)
        allowed_extensions = set(config.get("document_processing.allowed_extensions", [".pdf", ".docx", ".txt"]))
        uploads_dir = config.get_paths()["uploads"]

        for file in files:
            try:
                # Validate file
                if not file.filename:
                    failed_files.append({"filename": "Unknown", "error": "No filename provided"})
                    continue

                file_extension = Path(file.filename).suffix.lower()
                if file_extension not in allowed_extensions:
                    failed_files.append({
                        "filename": file.filename,
                        "error": f"Unsupported file type: {file_extension}"
                    })
                    continue

                # Check file size
                content = await file.read()
                if len(content) > max_file_size:
                    failed_files.append({
                        "filename": file.filename,
                        "error": f"File too large (max {max_file_size/1024/1024:.1f}MB)"
                    })
                    continue

                # Save file
                file_path = uploads_dir / file.filename
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(content)

                uploaded_files.append(str(file_path))
                logger.info(f"Uploaded file: {file.filename}")

            except Exception as e:
                logger.error(f"Error uploading file {file.filename}: {e}")
                failed_files.append({"filename": file.filename, "error": str(e)})

        if not uploaded_files:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "No files were successfully uploaded",
                    "failed_files": failed_files
                }
            )

        # Process uploaded documents
        processing_results = ai_agent.process_documents(uploaded_files)

        return JSONResponse(content={
            "message": f"Successfully processed {len(processing_results['successful'])} documents",
            "successful_files": processing_results['successful'],
            "failed_files": processing_results['failed'] + [f["filename"] for f in failed_files],
            "total_chunks": processing_results['total_chunks']
        })

    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question based on uploaded documents."""
    import time
    start_time = time.time()

    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Get answer from AI agent
        result = ai_agent.ask_question(request.question.strip())

        # Add processing time and model info
        processing_time = time.time() - start_time
        result["processing_time"] = round(processing_time, 2)
        result["model_used"] = config.get("ollama.model", "gemma:8b")

        return QuestionResponse(**result)

    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-with-payload")
async def ask_question_with_payload(payload: Dict[str, Any] = Body(...)):
    """Ask a question with custom configuration payload."""
    import time
    start_time = time.time()

    try:
        # Extract question and options from payload
        question = payload.get("question", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question is required in payload")

        # Temporarily update configuration if provided
        temp_config = payload.get("config", {})
        if temp_config:
            # Create a backup of current config
            original_config = config._config.copy()

            try:
                # Update configuration
                update_config_from_payload(temp_config)

                # Reinitialize AI agent with new config if model changed
                if "ollama" in temp_config:
                    global ai_agent
                    ai_agent = DocumentAIAgent()

                # Get answer with updated config
                result = ai_agent.ask_question(question)

            finally:
                # Restore original configuration
                config._config = original_config
                ai_agent = DocumentAIAgent()  # Reinitialize with original config
        else:
            # Use current configuration
            result = ai_agent.ask_question(question)

        # Add metadata
        processing_time = time.time() - start_time
        result.update({
            "processing_time": round(processing_time, 2),
            "model_used": config.get("ollama.model", "gemma:8b"),
            "config_used": temp_config if temp_config else "default"
        })

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error in ask-with-payload endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/update")
async def update_configuration(request: ConfigUpdateRequest):
    """Update system configuration from JSON payload."""
    try:
        # Validate configuration
        new_config = request.config

        # Update configuration
        update_config_from_payload(new_config)

        # Reinitialize components if necessary
        if "ollama" in new_config:
            global ai_agent
            ai_agent = DocumentAIAgent()
            logger.info("AI agent reinitialized with new configuration")

        return JSONResponse(content={
            "message": "Configuration updated successfully",
            "updated_sections": list(new_config.keys()),
            "current_config": config._config
        })

    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_configuration():
    """Get current system configuration."""
    try:
        return JSONResponse(content={
            "config": config._config,
            "paths": {name: str(path) for name, path in config.get_paths().items()},
            "status": "active"
        })

    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/manage")
async def manage_models(request: ModelRequest):
    """Manage Ollama models (pull, delete, info)."""
    try:
        model_name = request.model_name
        action = request.action.lower()

        if action == "pull":
            success = model_manager.pull_model(model_name)
            if success:
                return JSONResponse(content={
                    "message": f"Model {model_name} pulled successfully",
                    "action": "pull",
                    "model": model_name
                })
            else:
                raise HTTPException(status_code=500, detail=f"Failed to pull model {model_name}")

        elif action == "delete":
            success = model_manager.delete_model(model_name)
            if success:
                return JSONResponse(content={
                    "message": f"Model {model_name} deleted successfully",
                    "action": "delete",
                    "model": model_name
                })
            else:
                raise HTTPException(status_code=500, detail=f"Failed to delete model {model_name}")

        elif action == "info":
            model_info = model_manager.get_model_info(model_name)
            if model_info:
                return JSONResponse(content={
                    "model_info": model_info,
                    "action": "info",
                    "model": model_name
                })
            else:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}")

    except Exception as e:
        logger.error(f"Error managing model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/list")
async def list_models():
    """List available Ollama models."""
    try:
        models = model_manager.list_models()
        return JSONResponse(content={
            "models": models,
            "count": len(models),
            "current_model": config.get("ollama.model", "gemma:8b")
        })

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=StatusResponse)
async def get_system_status():
    """Get comprehensive system status."""
    try:
        # Check Ollama status
        ollama_status = validate_ollama_setup()

        # Get agent statistics
        agent_stats = ai_agent.get_stats()

        # Get configuration summary
        config_summary = {
            "app_name": config.get("app.name"),
            "debug_mode": config.get("app.debug"),
            "ollama_model": config.get("ollama.model"),
            "embedding_model": config.get("embedding.model"),
            "vector_store": config.get("vector_store.type"),
            "features_enabled": config.get("features", {})
        }

        # Determine overall status
        if ollama_status["status"] == "success":
            overall_status = "healthy"
        elif ollama_status["status"] == "warning":
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        return StatusResponse(
            status=overall_status,
            ollama_status=ollama_status,
            agent_stats=agent_stats,
            config_summary=config_summary
        )

    except Exception as e:
        logger.error(f"Error in status endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
async def clear_documents():
    """Clear all processed documents."""
    try:
        success = ai_agent.clear_knowledge_base()

        # Also clear uploaded files
        uploads_dir = config.get_paths()["uploads"]
        if uploads_dir.exists():
            shutil.rmtree(uploads_dir)
            uploads_dir.mkdir(exist_ok=True)

        if success:
            return JSONResponse(content={"message": "Successfully cleared all documents"})
        else:
            raise HTTPException(status_code=500, detail="Failed to clear documents")

    except Exception as e:
        logger.error(f"Error in clear endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed diagnostics."""
    try:
        # Basic health
        health_status = {
            "status": "healthy",
            "timestamp": None,
            "checks": {}
        }

        import datetime
        health_status["timestamp"] = datetime.datetime.utcnow().isoformat()

        # Check Ollama
        ollama_check = validate_ollama_setup()
        health_status["checks"]["ollama"] = ollama_check

        # Check file system
        paths = config.get_paths()
        health_status["checks"]["filesystem"] = {
            "data_dir": paths["data"].exists(),
            "uploads_dir": paths["uploads"].exists(),
            "writable": True  # Could add actual write test
        }

        # Check AI agent
        try:
            agent_stats = ai_agent.get_stats()
            health_status["checks"]["ai_agent"] = {
                "initialized": True,
                "stats": agent_stats
            }
        except Exception as e:
            health_status["checks"]["ai_agent"] = {
                "initialized": False,
                "error": str(e)
            }

        # Determine overall status
        if ollama_check["status"] == "error":
            health_status["status"] = "unhealthy"
        elif ollama_check["status"] == "warning":
            health_status["status"] = "degraded"

        status_code = 200 if health_status["status"] == "healthy" else 503

        return JSONResponse(
            content=health_status,
            status_code=status_code
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e)
            },
            status_code=503
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info(f"Starting {config.get('app.name')} v{config.get('app.version')}")

    # Validate Ollama setup
    ollama_status = validate_ollama_setup()
    if ollama_status["status"] == "error":
        logger.warning("Ollama validation failed, but continuing startup")
        logger.warning(f"Errors: {ollama_status['errors']}")
    else:
        logger.info("✅ Ollama validation successful")

    # Log configuration summary
    logger.info(f"Configuration: {config.get('ollama.model')} @ {config.get('ollama.base_url')}")

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Document AI Agent server...")
    logger.info(f"Server will be available at http://{config.host}:{config.port}")

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.get("server.reload", True),
        log_level=config.get("app.log_level", "info").lower()
    )
