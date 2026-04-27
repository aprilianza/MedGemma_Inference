# Lumira AI Enhanced — MedGemma Inference Service (GGUF)

Backend API untuk konsultasi medis multimodal yang ditenagai oleh **MedGemma-4B-IT-GGUF**. 
Telah dioptimasi untuk berjalan dengan VRAM efisien di server GPU standar maupun Apple Silicon (M-Series Mac) menggunakan akselerasi Metal (MPS).

## Arsitektur

```
Doctor / Patient Input (Teks + Base64 Image)
        │
        ▼
  FastAPI Server (main.py)
        │
        ▼
  MedGemmaInference (Singleton)
   ├── llama-cpp-python (Backend Inference Engine)
   ├── Llava15ChatHandler (Vision Projector mmproj)
   └── MedGemma-4B-IT-Q4_K_M.gguf (LLM Weights)
        │
        ▼
  JSON Response (API V2 Contract)
```

## Persiapan Instalasi

1. **Buat Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Llama-CPP (Khusus Mac Apple Silicon):**
   *(Gunakan perintah ini agar inference berjalan di GPU Mac, bukan CPU)*
   ```bash
   CMAKE_ARGS="-DGGML_METAL=on" pip install --no-cache-dir llama-cpp-python
   ```

4. **Login Hugging Face:**
   *(Diperlukan untuk akses model MedGemma GGUF dari HuggingFace Hub)*
   ```bash
   huggingface-cli login
   ```

## Menjalankan Server

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
*(Catatan: Saat dijalankan pertama kali, aplikasi akan otomatis mengunduh file `.gguf` seukuran ~3GB ke dalam cache komputer Anda).*

## Spesifikasi Endpoint API (V2)

### `POST /api/v2/consultation`

**Request Payload:**
```json
{
  "user_prompt": "Tolong analisis hasil USG payudara ini.",
  "user_role": "dokter", 
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSk...",
  "chat_history": []
}
```
*Keterangan Role:*
- `dokter`: AI akan merespon dengan bahasa klinis, terminologi medis (misal: BI-RADS), dan teknis.
- `pasien`: AI akan merespon dengan bahasa ramah, awam, penuh empati, dan tidak menegakkan diagnosis.

**Response:**
```json
{
  "status": "success",
  "message": "Konsultasi berhasil diproses.",
  "data": {
    "consultation_result": "Berdasarkan gambar USG payudara yang diberikan...",
    "profiling": {
      "latency_seconds": 8.81,
      "tokens_generated": 90,
      "tokens_per_second": 10.21
    }
  }
}
```

## Keamanan & Kontribusi
Proyek ini berisi *system prompt* untuk kepentingan Lumira AI. Pastikan tidak men-commit file gambar medis pasien asli ke dalam repositori ini.
