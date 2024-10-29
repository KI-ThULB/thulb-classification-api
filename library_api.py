from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import logging
import aiohttp
from datetime import datetime

from copilot_sacherschliessung_final import (
    Settings, RVKClient, RVKThULBMapper, DNBProcessor, BookMetadata, setup_logging
)

# API-Konfiguration
USE_API_KEY = False  # Entwicklungsmodus

app = FastAPI(
    title="Library Classification API",
    description="API für die automatische Sacherschließung von Büchern",
    version="1.0.0"
)

# CORS-Middleware hinzufügen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class ISBNRequest(BaseModel):
    isbn: str

class ISBNBatchRequest(BaseModel):
    isbns: List[str]

# Globale Komponenten
settings = Settings()
rvk_client = RVKClient(settings)
mapper = RVKThULBMapper()
processor = None

# API-Key Validierung
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if USE_API_KEY:
        if x_api_key is None or x_api_key != "your-api-key":
            raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key

@app.on_event("startup")
async def startup_event():
    """Initialisiert die notwendigen Komponenten beim Start."""
    global processor
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.debug("Starting up API with debug logging")
    
    processor = DNBProcessor(settings, rvk_client, mapper)
    await processor.__aenter__()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup beim Herunterfahren."""
    if processor:
        await processor.__aexit__(None, None, None)

@app.get("/health")
async def health_check():
    """API Health Check"""
    logger = logging.getLogger(__name__)
    try:
        # Test DNB-Verbindung
        async with aiohttp.ClientSession() as session:
            async with session.get(str(settings.DNB_SRU_URL)) as response:
                dnb_available = response.status == 200
                logger.debug(f"DNB connection status: {response.status}")
                
        # Test RVK-Verbindung
        async with aiohttp.ClientSession() as session:
            async with session.get(str(settings.RVK_API_URL)) as response:
                rvk_available = response.status == 200
                logger.debug(f"RVK connection status: {response.status}")
                
        status = "healthy" if (dnb_available and rvk_available) else "degraded"
        
        return {
            "status": status,
            "components": {
                "dnb_connection": dnb_available,
                "rvk_connection": rvk_available,
                "processor_initialized": processor is not None
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/process-isbn/")
async def process_single_isbn(
    request: ISBNRequest,
    api_key: str = Depends(verify_api_key) if USE_API_KEY else None
):
    """Verarbeitet eine einzelne ISBN mit verbesserter Fehlerbehandlung."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Processing single ISBN: {request.isbn}")
    
    try:
        # Prüfe, ob Processor initialisiert ist
        if not processor:
            raise HTTPException(
                status_code=503,
                detail="Service nicht initialisiert. Bitte später erneut versuchen."
            )
            
        # Validiere ISBN
        if not processor._validate_isbn(request.isbn):
            raise HTTPException(
                status_code=400,
                detail=f"Ungültige ISBN: {request.isbn}"
            )
            
        # Health Check
        health_status = await health_check()
        if health_status["status"] == "unhealthy":
            raise HTTPException(
                status_code=503,
                detail="Service temporär nicht verfügbar"
            )
            
        # Verarbeite ISBN
        result = await processor.process_isbn(request.isbn)
        if result:
            logger.debug(f"Found results for ISBN {request.isbn}")
            return result.to_dict()
            
        # Keine Ergebnisse
        raise HTTPException(
            status_code=404,
            detail=f"Keine Ergebnisse für ISBN {request.isbn} gefunden"
        )
            
    except aiohttp.ClientError as e:
        logger.error(f"Connection error processing ISBN {request.isbn}: {e}")
        raise HTTPException(
            status_code=503,
            detail="Externe Dienste (DNB/RVK) temporär nicht verfügbar"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing ISBN {request.isbn}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Interner Server-Fehler: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
