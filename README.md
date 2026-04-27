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
   _(Gunakan perintah ini agar inference berjalan di GPU Mac, bukan CPU)_

   ```bash
   CMAKE_ARGS="-DGGML_METAL=on" pip install --no-cache-dir llama-cpp-python
   ```

4. **Login Hugging Face:**
   _(Diperlukan untuk akses model MedGemma GGUF dari HuggingFace Hub)_
   ```bash
   huggingface-cli login
   ```

## Menjalankan Server

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

_(Catatan: Saat dijalankan pertama kali, aplikasi akan otomatis mengunduh file `.gguf` seukuran ~3GB ke dalam cache komputer Anda)._

## Deploy VPS Dengan Docker Compose

File `Dockerfile` dan `docker-compose.yml` sudah disiapkan agar service bisa langsung jalan di VPS.

1. **Siapkan token Hugging Face (wajib):**

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

2. **Build dan jalankan service:**

```bash
docker compose up -d --build
```

3. **Cek status container:**

```bash
docker compose ps
docker compose logs -f medgemma-api
```

4. **Akses API:**

- Health check: `GET http://<IP_VPS>:8000/health`
- Inference API: `POST http://<IP_VPS>:8000/consultations`

Catatan penting:

- Volume `hf_cache` dipakai untuk menyimpan cache model Hugging Face agar tidak download ulang setiap restart/deploy.
- Startup pertama bisa memakan waktu lebih lama karena unduh model GGUF + mmproj.
- Pastikan firewall VPS membuka port `8000`.

## Spesifikasi Endpoint API (V2)

### `POST /consultations`

**Request Payload:**

```json
{
  "user_prompt": "Tolong analisis hasil USG payudara ini.",
  "user_role": "dokter",
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSk...",
  "chat_history": []
}
```

_Keterangan Role:_

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

Proyek ini berisi _system prompt_ untuk kepentingan Lumira AI. Pastikan tidak men-commit file gambar medis pasien asli ke dalam repositori ini.
