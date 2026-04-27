import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from MedGemma_Inference import MedGemmaInference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Singleton reference, diinisialisasi saat startup ──
med_gemma: MedGemmaInference | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model sekali saat startup, release saat shutdown."""
    global med_gemma
    logger.info("Inisialisasi MedGemma model...")
    med_gemma = MedGemmaInference()
    logger.info("Server siap menerima request.")
    yield
    logger.info("Shutting down server.")


app = FastAPI(
    title="Lumira AI Enhanced — MedGemma Inference Service",
    version="2.0.0",
    description="Multimodal medical consultation API powered by MedGemma-4B-IT",
    lifespan=lifespan,
)


# ── Request / Response Schemas ──
class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' atau 'assistant'")
    content: str = Field(..., description="Isi pesan")


class ConsultationRequest(BaseModel):
    user_prompt: str = Field(..., description="Pertanyaan / instruksi dokter atau pasien", min_length=1)
    user_role: str = Field(default="dokter", description="Role pengguna: 'dokter' atau 'pasien'")
    chat_history: list[ChatMessage] | None = Field(
        default=None, description="Riwayat percakapan (maks 10 terakhir)"
    )
    image_url: str | None = Field(
        default=None, description="URL citra medis (USG, mammogram, dll.)"
    )
    image_base64: str | None = Field(
        default=None, description="String Base64 gambar medis (untuk upload lokal)"
    )


class ProfilingData(BaseModel):
    latency_seconds: float
    tokens_generated: int
    tokens_per_second: float


class ConsultationData(BaseModel):
    consultation_result: str
    profiling: ProfilingData


class ConsultationResponse(BaseModel):
    status: str
    message: str
    data: ConsultationData | None


# ── Endpoints ──
@app.post("/api/v2/consultation", response_model=ConsultationResponse)
async def consultation(req: ConsultationRequest):
    """
    Endpoint utama untuk konsultasi medis multimodal.
    Mendukung teks + gambar (URL citra medis).
    """
    if med_gemma is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")

    # Konversi chat_history dari Pydantic ke dict
    history = None
    if req.chat_history:
        history = [m.model_dump() for m in req.chat_history]

    # Jalankan inferensi di thread terpisah agar tidak memblokir event loop
    # + gunakan async lock agar hanya 1 inferensi berjalan (model bukan thread-safe)
    async with med_gemma.inference_lock:
        result = await asyncio.to_thread(
            med_gemma.generate_response,
            user_prompt=req.user_prompt,
            user_role=req.user_role,
            chat_history=history,
            image_url=req.image_url,
            image_base64=req.image_base64,
        )

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return result


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": med_gemma is not None and med_gemma._initialized,
    }
