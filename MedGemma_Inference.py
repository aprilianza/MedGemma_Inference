import time
import asyncio
import logging
import requests
import threading
import base64
from io import BytesIO
from PIL import Image
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

logger = logging.getLogger(__name__)

# Hapus SYSTEM_PROMPT statis dari sini karena kita akan membuatnya dinamis di dalam fungsi berdasarkan role.


class MedGemmaInference:
    """
    Singleton inference engine untuk MedGemma-4B-IT format GGUF.
    Menggunakan llama-cpp-python untuk CPU/GPU fallback (Apple Silicon MPS / CUDA).
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, repo_id: str = "SandLogicTechnologies/MedGemma-4B-IT-GGUF"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, repo_id: str = "SandLogicTechnologies/MedGemma-4B-IT-GGUF"):
        if self._initialized:
            return

        logger.info(f"Mendownload model GGUF dari {repo_id} (bila belum ada)...")

        # 1. Download model utama (Q4_K_M untuk efisiensi)
        self.model_path = hf_hub_download(
            repo_id=repo_id, 
            filename="medgemma-4b-it_Q4_K_M.gguf"
        )
        
        # 2. Download vision projector (mmproj) untuk input gambar
        self.mmproj_path = hf_hub_download(
            repo_id=repo_id, 
            filename="mmproj-medgemma-4b-it-F16.gguf"
        )

        logger.info("Inisialisasi Llama-CPP dengan Vision Handler...")
        
        # 3. Setup Vision Handler
        self.chat_handler = Llava15ChatHandler(clip_model_path=self.mmproj_path)
        
        # 4. Load Model (n_gpu_layers=-1 berarti full masuk GPU/VRAM)
        self.model = Llama(
            model_path=self.model_path,
            chat_handler=self.chat_handler,
            n_ctx=2048,           # Batas panjang konteks
            n_gpu_layers=-1,      # Offload seluruh layer ke GPU (Apple Silicon / CUDA)
            verbose=False
        )

        self.inference_lock = asyncio.Lock()
        self._initialized = True
        logger.info("Model MedGemma GGUF siap digunakan.")

    def _download_image_to_base64(self, image_url: str, timeout: int = 15) -> str | None:
        """Unduh gambar dan konversi ke Base64 (dibutuhkan oleh llama.cpp)."""
        try:
            # Tambahkan User-Agent agar request tidak diblokir oleh server (contoh: Wikimedia)
            headers = {"User-Agent": "LumiraAI-Backend/1.0 (Mozilla/5.0)"}
            resp = requests.get(image_url, headers=headers, timeout=timeout, stream=True)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content)).convert("RGB")
            
            # Konversi ke JPEG base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            logger.warning(f"Gagal mengunduh gambar: {e}")
            return None

    def _build_messages(self, user_prompt: str, user_role: str, chat_history: list | None, image_b64: str | None) -> list:
        """Menyusun format chat untuk llama.cpp dengan penyesuaian Role (Pasien/Dokter)."""
        
        # 1. Tentukan gaya bahasa (System Prompt) berdasarkan Role dengan paksaan Bahasa Indonesia
        if user_role.lower() == "patient":
            system_prompt = (
                "Anda adalah MedGemma, asisten medis AI yang penuh empati untuk platform Lumira AI Enhanced. "
                "Anda sedang berbicara langsung dengan PASIEN. Gunakan Bahasa Indonesia yang jelas, sopan, ramah, "
                "dan mudah dipahami. Jelaskan istilah medis dengan sangat sederhana. "
                "SELALU ingatkan pasien bahwa analisis Anda BUKAN diagnosis resmi dan mereka WAJIB periksa ke dokter. "
                "PERINTAH MUTLAK: Anda HARUS selalu menjawab menggunakan Bahasa Indonesia."
            )
        else:
            system_prompt = (
                "Anda adalah MedGemma, asisten medis AI untuk platform Lumira AI Enhanced. "
                "Anda sedang membantu seorang DOKTER/TENAGA KLINIS. Berikan analisis medis yang detail, "
                "objektif, presisi, dan profesional menggunakan terminologi medis yang tepat. "
                "PERINTAH MUTLAK: Anda HARUS selalu menjawab menggunakan Bahasa Indonesia."
            )

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if chat_history:
            for msg in chat_history[-10:]:
                messages.append({"role": msg["role"], "content": msg.get("content", "")})

        # Format multimodal llama-cpp: content berupa list berisi text dan image_url
        if image_b64:
            user_content = [
                {"type": "image_url", "image_url": {"url": image_b64}},
                {"type": "text", "text": f"[Gambar Medis Terlampir]\n{user_prompt}\n\n[PENTING: Jawab seluruh analisis klinis di atas HANYA menggunakan Bahasa Indonesia!]"}
            ]
        else:
            user_content = user_prompt

        messages.append({"role": "user", "content": user_content})
        return messages

    def generate_response(
        self,
        user_prompt: str,
        user_role: str = "dokter",
        chat_history: list | None = None,
        image_url: str | None = None,
        image_base64: str | None = None,
    ) -> dict:
        """Inferensi menggunakan Llama-CPP backend."""
        start_time = time.time()

        try:
            image_b64 = None
            
            # Prioritaskan base64 langsung jika ada
            if image_base64:
                # Pastikan ada prefix MIME-Type untuk llama.cpp
                if not image_base64.startswith("data:image/"):
                    image_b64 = f"data:image/jpeg;base64,{image_base64}"
                else:
                    image_b64 = image_base64
            # Jika tidak ada base64, coba download dari URL
            elif image_url:
                image_b64 = self._download_image_to_base64(image_url)
                if not image_b64:
                    return {
                        "status": "error",
                        "message": f"Gagal mengunduh gambar dari URL: {image_url}",
                        "data": None,
                    }

            messages = self._build_messages(user_prompt, user_role, chat_history, image_b64)

            # Eksekusi inferensi
            output = self.model.create_chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.3, # Dinaikkan sedikit agar tidak kaku
                repeat_penalty=1.15, # MENCEGAH MODEL MENGULANG-ULANG KALIMAT
                # Menambahkan ASSISTANT: dan <end_of_turn> agar LLM berhenti bicara setelah paragraf pertama selesai
                stop=["User:", "Doctor:", "Patient:", "\n\n\n", "ASSISTANT:", "USER:", "<end_of_turn>"], 
                stream=False
            )

            response_text = output["choices"][0]["message"]["content"].strip()
            
            # Post-processing manual: Potong output jika masih membocorkan kata ASSISTANT:
            if "ASSISTANT:" in response_text:
                response_text = response_text.split("ASSISTANT:")[0].strip()
            
            # Hitung metrics
            latency = time.time() - start_time
            token_count = output["usage"]["completion_tokens"]
            tps = token_count / latency if latency > 0 else 0

            return {
                "status": "success",
                "message": "Konsultasi berhasil diproses.",
                "data": {
                    "consultation_result": response_text,
                    "profiling": {
                        "latency_seconds": round(latency, 2),
                        "tokens_generated": token_count,
                        "tokens_per_second": round(tps, 2),
                    },
                },
            }

        except Exception as e:
            logger.exception("Inference error")
            return {
                "status": "error",
                "message": f"Inferensi gagal: {str(e)}",
                "data": None,
            }