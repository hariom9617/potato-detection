"""
FastAPI application entry point.
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging

from .config import settings
from .core import setup_logging, AppException
from .api.v1 import api_router
from .schemas.health import HealthResponse


# Setup logging
setup_logging(
    level=settings.log_level,
    log_file=settings.log_file,
)

logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="AI-powered potato quality inspection API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# Exception handlers

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """Handle application exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(request_id=getattr(request.state, "request_id", None))
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        }
    )


# Root health endpoint (unversioned)

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Basic Health Check",
    description="Simple health check endpoint (always returns healthy if service is running)",
)
async def health_check():
    """Basic health check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.api_version,
    )


# Include v1 API router
app.include_router(api_router, prefix="/api/v1")


# Startup/shutdown events

@app.on_event("startup")
async def startup_event():
    """Application startup."""
    logger.info(f"Starting {settings.api_title} v{settings.api_version}")
    logger.info(f"Environment: {'Development' if settings.api_reload else 'Production'}")
    logger.info(f"CORS origins: {settings.cors_origins}")
    logger.info(f"Model path: {settings.model_path or 'Not configured'}")
    logger.info(f"Device: {settings.device}")
    logger.info(f"Database persistence: {settings.enable_persistence}")
    
    # Ensure directories exist
    settings.ensure_directories()
    logger.info("Required directories verified")
    
    # Initialize orchestrator to check model loading
    from .api.deps import get_orchestrator
    
    try:
        orchestrator = get_orchestrator()
        model_statuses = orchestrator.get_model_statuses()
        
        available_models = [m for m in model_statuses if m["status"] == "available"]
        logger.info(f"Models available: {len(available_models)}/{len(model_statuses)}")
        
        for model in model_statuses:
            status_msg = f"  - {model['name']}: {model['status']}"
            if model.get('reason'):
                status_msg += f" ({model['reason']})"
            logger.info(status_msg)
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        logger.warning("Application will continue but model inference may not work")
    
    logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown."""
    logger.info("Shutting down application")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
