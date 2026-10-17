# Audio Dataset Pipeline (MSc Dissertation)

**Peace Ojehanon** · Glasgow, UK · [LinkedIn](https://www.linkedin.com/in/peace-ojehanon) · peaceojehanon5@gmail.com

The data pipeline behind my MSc dissertation, *"Developing a High-Quality
Database for Synthesized Audio Extracted from Video Sources to Enhance
Real-Time Virtual Communication"* (Glasgow Caledonian University, 2025). This
was my own individual project.

It searches YouTube by topic through the YouTube Data API, downloads the audio
only, cleans it and transcribes it, then stores everything on Google Cloud:
raw and cleaned audio in Cloud Storage, and transcripts and video metadata in
BigQuery. I then measured how much the cleaning actually helped.

## Pipeline stages

```
topics → YouTube Data API v3 search → download audio only (yt-dlp)
       → convert to MP3 (FFmpeg)
       → clean: noise reduction (noisereduce) + loudness normalisation (pydub)
       → transcribe (Google Cloud Speech-to-Text, English)
       → store: audio in Cloud Storage, transcripts and metadata in BigQuery
       → evaluate: SNR (global and segmental), Mel cepstral distortion, spectrograms
```

## Making it reliable

- **Safe to rerun.** Before downloading or cleaning anything, the pipeline
  checks what is already in the Cloud Storage bucket and skips files it has
  already processed, so a rerun tops up the dataset instead of fetching it all
  again.
- **Retries.** Temporary Speech-to-Text API errors are retried with
  exponential back-off.
- **Schema changes.** Any missing BigQuery columns are added before each insert.
- **Data-quality check.** In the dissertation I also checked the BigQuery table
  with a SQL query (missing values, transcript lengths and topic balance). That
  query is in the report's appendix and isn't in this repo yet.

## Evaluation

I compared each cleaned clip with its raw version using signal-to-noise ratio
(global and segmental) and Mel cepstral distortion, and looked at spectrograms
of both. The result was an average SNR improvement of 1.73 dB across 10
sampled clips. The gains were modest and uneven, with the noisier clips gaining
most, and the dissertation reports them that way.

## Files

| File | What it is |
|------|------------|
| `pipeline.py` | The full pipeline, followed by the evaluation steps (SNR, segmental SNR, spectrograms and Mel cepstral distortion). Exported from the original Google Colab notebook. |
| `evaluation.py` | A separate evaluation notebook, also exported from Colab: global and segmental SNR for every cleaned clip in the bucket. |

## Configuration and credentials

No API keys or credentials are committed. The scripts read them from
environment variables:

```bash
export YOUTUBE_API_KEY="your-youtube-data-api-key"
export GCP_PROJECT_ID="your-gcp-project-id"
export GCS_BUCKET="your-bucket-name"      # optional, has a default
export BQ_DATASET="your-bigquery-dataset" # optional, has a default

gcloud auth application-default login
gcloud auth application-default set-quota-project "$GCP_PROJECT_ID"
```

## Run it

Both files are exports of Google Colab notebooks and still contain notebook
shell lines (`!pip install ...`, `!gcloud auth ...`), so they won't run as
plain `python` scripts. Paste them into a Colab or Jupyter notebook and run
them there after setting the environment variables above. The `!pip` lines
install the Python packages; FFmpeg must also be available on the machine.

You'll need a Google Cloud project with the Cloud Storage, Speech-to-Text and
BigQuery APIs enabled, and a YouTube Data API key.

## Scope

Real-time use was out of scope. The dissertation built and evaluated the
dataset pipeline, not a live communication system.

## Tech

Python · Google Cloud (Cloud Storage, Speech-to-Text, BigQuery) · YouTube Data
API v3 · yt-dlp · FFmpeg · noisereduce · pydub · librosa.
