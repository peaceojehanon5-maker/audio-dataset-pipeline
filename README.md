# Audio Dataset Pipeline (MSc Dissertation)

**Peace Ojehanon** · Glasgow, UK · [LinkedIn](https://www.linkedin.com/in/peace-ojehanon) · peaceojehanon5@gmail.com

The data pipeline behind my MSc dissertation, *"Developing a High-Quality
Database for Synthesized Audio Extracted from Video Sources to Enhance
Real-Time Virtual Communication"* (Glasgow Caledonian University, 2025).

It takes a list of topics, finds matching videos, pulls clean audio out of
them, transcribes it, and lands everything — audio in cloud storage, metadata
and transcripts in a queryable warehouse — as a reusable dataset. The quality
of the extracted audio is then **measured**, not assumed.

> **A related, fully reproducible companion** lives in my
> [Audio Processing Portfolio](https://github.com/peaceojehanon5-maker/Audio-Video-Production-Specialists),
> which rebuilds this DSP toolkit from first principles on synthesised signals
> (so the SNR figures can be checked against a known ground truth).

## Pipeline stages

```
topics ─▶ YouTube Data API search ─▶ download (yt-dlp)
       ─▶ audio extraction (FFmpeg)
       ─▶ cleaning: noise reduction (noisereduce) + resample to 44.1 kHz (librosa)
       ─▶ transcription (Google Speech-to-Text)
       ─▶ store: audio → Cloud Storage,  metadata + transcripts → BigQuery
       ─▶ evaluate: SNR (global + segmental), PESQ, STOI
```

Two design points worth calling out:

- **Idempotent ingestion** — before downloading, the pipeline checks what's
  already in the Cloud Storage bucket and skips duplicates, so a re-run tops up
  the dataset instead of re-fetching it.
- **Honest evaluation** — extracted audio is scored with established perceptual
  and signal metrics (PESQ, STOI, segmental SNR) rather than eyeballed.

## Files

| File | What it is |
|------|------------|
| `pipeline.py` | The end-to-end pipeline (adapted from the original Colab notebook). |
| `evaluation.py` | Standalone quality evaluation — SNR / PESQ / STOI over the stored clips. |

## Configuration & credentials

Nothing sensitive is committed. The scripts read everything from environment
variables:

```bash
export YOUTUBE_API_KEY="your-youtube-data-api-key"
export GCP_PROJECT_ID="your-gcp-project-id"
export GCS_BUCKET="your-bucket-name"      # optional, has a default
export BQ_DATASET="your-bigquery-dataset" # optional, has a default

gcloud auth application-default login
gcloud auth application-default set-quota-project "$GCP_PROJECT_ID"
```

## Run it

```bash
pip install -U yt-dlp ffmpeg-python librosa noisereduce soundfile pydub \
    google-cloud-storage google-cloud-speech google-cloud-bigquery \
    pesq pystoi tabulate matplotlib
# FFmpeg must be installed on the system (https://ffmpeg.org)

python3 pipeline.py      # build / top up the dataset
python3 evaluation.py    # score audio quality
```

You'll need a Google Cloud project with the Cloud Storage, Speech-to-Text and
BigQuery APIs enabled, and a YouTube Data API key.

## Scope

The dissertation built and evaluated the **dataset pipeline**. Live, real-time
integration into a communication system was deliberately out of scope and noted
as future work — the contribution here is the high-quality, measured,
queryable audio dataset and the repeatable process that produces it.

## Tech

Python · Google Cloud (Cloud Storage, Speech-to-Text, BigQuery) · yt-dlp ·
FFmpeg · librosa · noisereduce · PESQ · STOI.
