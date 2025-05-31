from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import gc
import psutil

# Importar solo cuando sea necesario para ahorrar memoria
try:
    from anime_recommender import AnimeRecommender
except ImportError:
    print("❌ Error importando AnimeRecommender")
    AnimeRecommender = None

# Inicializar FastAPI
app = FastAPI(
    title="Anime Recommendation API",
    description="Sistema de recomendación de anime basado en contenido",
    version="1.0.0"
)

# Configuración CORS mejorada
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://anime-ratings-analysis.onrender.com",
]

# En producción, usar solo origins específicos
if os.getenv("ENVIRONMENT") == "production":
    origins = [
        "https://anime-ratings-analysis.onrender.com",
        "https://tu-frontend-domain.com",  # Cambiar por tu dominio real
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.getenv("ENVIRONMENT") != "production" else origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "HEAD"],
    allow_headers=["*"],
)

# Global recommender instance
recommender = None
model_loading_error = None

# Pydantic models
class RecommendationRequest(BaseModel):
    anime_name: str
    num_recommendations: Optional[int] = 10

class RecommendationItem(BaseModel):
    rank: int
    name: str
    rating: float
    genre: str
    similarity_score: float

class RecommendationResponse(BaseModel):
    success: bool
    message: str
    data: List[RecommendationItem]
    total_recommendations: int

class AnimeListResponse(BaseModel):
    success: bool
    message: str
    animes: List[str]
    total_count: int

# Configuración del modelo con variable de entorno
MODEL_CONFIG = {
    "google_drive_file_id": os.getenv("MODEL_FILE_ID", "TU_FILE_ID_AQUI"),
    "model_path": "anime_model.pkl"
}

def get_memory_usage():
    """Obtiene el uso actual de memoria"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Memoria física
            "vms_mb": memory_info.vms / 1024 / 1024,  # Memoria virtual
        }
    except:
        return {"rss_mb": 0, "vms_mb": 0}

# Startup event optimizado
@app.on_event("startup")
async def startup_event():
    global recommender, model_loading_error
    
    print("🚀 Iniciando servidor...")
    print(f"💾 Memoria inicial: {get_memory_usage()}")
    
    # Verificar disponibilidad de la clase
    if AnimeRecommender is None:
        model_loading_error = "AnimeRecommender no disponible"
        print(f"❌ {model_loading_error}")
        return
    
    try:
        recommender = AnimeRecommender()
        
        # Verificar configuración del FILE_ID
        file_id = MODEL_CONFIG["google_drive_file_id"]
        model_path = MODEL_CONFIG["model_path"]
        
        if file_id == "TU_FILE_ID_AQUI" or not file_id:
            model_loading_error = "FILE_ID no configurado correctamente"
            print("⚠️  ADVERTENCIA: MODEL_FILE_ID no configurado!")
            print("💡 Configura MODEL_FILE_ID como variable de entorno en Render")
            print("🔗 Ejemplo: MODEL_FILE_ID=1abc123def456ghi789...")
            return
        
        print(f"📁 Usando FILE_ID: {file_id[:20]}...")
        
        # Verificar memoria antes de descargar
        memory_before = get_memory_usage()
        print(f"💾 Memoria antes de descarga: {memory_before['rss_mb']:.1f}MB")
        
        # Intentar descargar modelo si no existe
        if not os.path.exists(model_path):
            print("🔄 Descargando modelo desde Google Drive...")
            success = recommender.download_model(file_id, model_path)
            if not success:
                model_loading_error = "Error descargando modelo desde Google Drive"
                print(f"❌ {model_loading_error}")
                return
        else:
            print("📂 Modelo ya existe localmente")
        
        # Verificar tamaño del archivo
        try:
            file_size_mb = os.path.getsize(model_path) / 1024 / 1024
            print(f"📊 Tamaño del modelo: {file_size_mb:.1f}MB")
            
            # Advertir si el archivo es muy grande para el plan gratuito
            if file_size_mb > 300:  # Conservador para 512MB RAM
                print("⚠️  ADVERTENCIA: Modelo muy grande para plan gratuito")
                print("💡 Considera optimizar el modelo o upgradar a plan pagado")
        except:
            pass
        
        # Cargar modelo con manejo de memoria
        print("📚 Cargando modelo en memoria...")
        memory_before_load = get_memory_usage()
        
        success = recommender.load_model(model_path)
        
        if success:
            memory_after = get_memory_usage()
            print("✅ Modelo cargado exitosamente")
            print(f"📊 Animes disponibles: {len(recommender.get_anime_list())}")
            print(f"💾 Memoria después de carga: {memory_after['rss_mb']:.1f}MB")
            print(f"📈 Memoria usada por modelo: {memory_after['rss_mb'] - memory_before_load['rss_mb']:.1f}MB")
            
            # Limpiar memoria no usada
            gc.collect()
            
        else:
            model_loading_error = "Error cargando modelo en memoria"
            print(f"❌ {model_loading_error}")
            
            # Limpiar archivo corrupto
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    print("🧹 Archivo corrupto eliminado")
                except Exception as cleanup_error:
                    print(f"⚠️  No se pudo eliminar archivo: {cleanup_error}")
    
    except Exception as e:
        model_loading_error = f"Error en startup: {str(e)}"
        print(f"❌ {model_loading_error}")
        recommender = None

# Routes
@app.get("/")
async def root():
    return {
        "message": "Anime Recommendation API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": ["/animes", "/recommend", "/health"],
        "memory_usage": get_memory_usage()
    }

@app.head("/")
async def root_head():
    """Handle HEAD requests for health checks"""
    return {}

@app.get("/health")
async def health_check():
    """Health check endpoint mejorado"""
    model_loaded = recommender is not None and recommender.sig_matrix is not None
    
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "loading_error": model_loading_error,
        "total_animes": len(recommender.get_anime_list()) if model_loaded else 0,
        "memory_usage": get_memory_usage(),
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/animes", response_model=AnimeListResponse)
async def get_anime_list():
    """Obtiene lista completa de animes disponibles"""
    if recommender is None or recommender.sig_matrix is None:
        error_msg = model_loading_error or "Modelo no disponible"
        raise HTTPException(
            status_code=503, 
            detail=f"Servicio no disponible: {error_msg}"
        )
    
    try:
        anime_list = recommender.get_anime_list()
        return AnimeListResponse(
            success=True,
            message="Lista de animes obtenida exitosamente",
            animes=anime_list,
            total_count=len(anime_list)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Genera recomendaciones para un anime específico"""
    if recommender is None or recommender.sig_matrix is None:
        error_msg = model_loading_error or "Modelo no disponible"
        raise HTTPException(
            status_code=503, 
            detail=f"Servicio no disponible: {error_msg}"
        )
    
    # Validar número de recomendaciones
    if request.num_recommendations < 1 or request.num_recommendations > 20:  # Reducido para ahorrar memoria
        raise HTTPException(
            status_code=400, 
            detail="num_recommendations debe estar entre 1 y 20"
        )
    
    try:
        result = recommender.recommend(
            request.anime_name, 
            n_recommendations=request.num_recommendations
        )
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["message"])
        
        # Convertir a formato de respuesta
        recommendations = [
            RecommendationItem(**item) for item in result["data"]
        ]
        
        return RecommendationResponse(
            success=True,
            message=result["message"],
            data=recommendations,
            total_recommendations=len(recommendations)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/anime/{anime_name}")
async def get_anime_info(anime_name: str):
    """Obtiene información específica de un anime"""
    if recommender is None or recommender.anime_data is None:
        error_msg = model_loading_error or "Modelo no disponible"
        raise HTTPException(
            status_code=503, 
            detail=f"Servicio no disponible: {error_msg}"
        )
    
    try:
        anime_info = recommender.anime_data[
            recommender.anime_data["name"].str.contains(anime_name, case=False)
        ]
        
        if anime_info.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"Anime '{anime_name}' no encontrado"
            )
        
        # Tomar el primer resultado si hay múltiples matches
        anime = anime_info.iloc[0]
        
        return {
            "success": True,
            "data": {
                "name": anime["name"],
                "rating": float(anime["rating"]),
                "genre": anime["genre"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint no encontrado", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Error interno del servidor", "status_code": 500}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)