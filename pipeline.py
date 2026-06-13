# -*- coding: utf-8 -*-
"""
Audio dataset pipeline — MSc dissertation (Glasgow Caledonian University, 2025).
Peace Ojehanon.

Credentials and project identifiers are read from environment variables
(YOUTUBE_API_KEY, GCP_PROJECT_ID, GCS_BUCKET, BQ_DATASET). Nothing secret
is committed. Adapted from the original Google Colab notebook.
"""

#Required installations:
!pip install -U yt-dlp
!pip install ffmpeg-python
!pip install google-cloud-storage google-cloud-speech google-cloud-bigquery
!pip install google-cloud-speech==2.18.0
!pip install -U librosa
!pip install librosa pydub noisereduce
!pip install soundfile
!gcloud auth application-default login
# set your quota project: gcloud auth application-default set-quota-project "$GCP_PROJECT_ID"
!pip install pesq
!pip install pystoi
!pip install tabulate # Install tabulate for table output
!pip install matplotlib # Install matplotlib for plotting

import os
import subprocess
import logging
from datetime import datetime

import yt_dlp
import ffmpeg
import librosa
import noisereduce as nr
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import tempfile
import gcsfs


from google.auth import default
from tabulate import tabulate
import matplotlib.pyplot as plt # Import matplotlib for plotting
from google.cloud import storage, bigquery
from google.cloud import speech
from googleapiclient.discovery import build
from google.api_core.retry import Retry
from google.cloud.speech import SpeechClient, RecognitionConfig, RecognitionAudio
from pesq import pesq
from pystoi import stoi

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

"""Global configuration"""
API_KEY = os.environ.get("YOUTUBE_API_KEY", "")  # Your YouTube API key
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")                         # Your GCP project id
BUCKET_NAME = os.environ.get("GCS_BUCKET", "youtube-videos-extracted")                # Your GCS bucket name
DATASET_ID = os.environ.get("BQ_DATASET", "my_dataset")                               # Your BigQuery dataset ID
TABLE_ID = "audio_metadata"                             # Your BigQuery table ID
TOPICS = [
    'Glasgow',
    'Student',
    'Visual Communication Platforms',
    'Big Data'
]
MAX_VIDEOS_PER_TOPIC = 6
LOCAL_DOWNLOAD_DIR = "downloads"
SAMPLE_RATE = 16000  # For transcription

os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)

# Initialize a global GCS client to avoid repeated instantiation
client_storage = storage.Client(project=PROJECT_ID)
# Initialize **one** Speech-to-Text client (reuse to avoid repeated handshakes)
speech_client = SpeechClient()

# Custom predicate function for retry logic
def retry_predicate(exception):
    """Returns True if the exception is retryable, False otherwise."""
    return isinstance(exception, (
        google.api_core.exceptions.DeadlineExceeded,
        google.api_core.exceptions.ServiceUnavailable,
        google.api_core.exceptions.InternalServerError,
        google.api_core.exceptions.TooManyRequests,
        google.api_core.exceptions.Aborted,
        google.api_core.exceptions.Unavailable,
    ))


# A retry policy: up to 3 total attempts, with exponential back-off starting at 1s.
speech_retry = Retry(
    initial=1.0,     # seconds first back-off
    maximum=10.0,    # seconds max back-off
    multiplier=2.0,  # double delay each retry
    deadline=30.0,   # give up entirely after 30s
    # Use Retry.with_predicate and static methods to define the predicate
     predicate=retry_predicate  # Use the custom predicate function
)

# -------------------------------------------------------------------
# Utility: List files in GCS by prefix (for duplicate checks)
# -------------------------------------------------------------------
def list_gcs_files(prefix):
    """
    Returns a set of full blob names under a given prefix in the bucket.
    """
    bucket = client_storage.bucket(BUCKET_NAME)
    files = {blob.name for blob in bucket.list_blobs(prefix=prefix)}
    logging.debug(f"Files in '{prefix}/': {files}")
    return files

def file_exists_in_gcs(file_name, prefix):
    """
    Checks if a file (by its exact name) already exists in a given GCS folder.
    """
    full_name = f"{prefix}/{file_name}"
    exists = full_name in list_gcs_files(prefix)
    logging.debug(f"Checking if {full_name} exists: {exists}")
    return exists

def upload_to_gcs(local_file, destination_blob):
    """
    Uploads a local file to GCS.
    """
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(local_file)
        logging.info(f"Uploaded {local_file} to gs://{BUCKET_NAME}/{destination_blob}")
    except Exception as e:
        logging.error(f"Error uploading {local_file} to GCS: {e}")

def download_from_gcs(source_blob, local_file):
    """
    Downloads a file from GCS to local storage.
    """
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(source_blob)
        blob.download_to_filename(local_file)
        logging.info(f"Downloaded gs://{BUCKET_NAME}/{source_blob} to {local_file}")
    except Exception as e:
        logging.error(f"Error downloading {source_blob} from GCS: {e}")

# -------------------------------
# Stage 2: YouTube Functions – Video Ingestion
# -------------------------------

def get_youtube_videos(api_key, query, max_results=5):
    """
    Fetches a list of videos (title, URL, metadata) from YouTube for a given query.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.search().list(
        q=query,
        part='id,snippet',
        maxResults=max_results,
        type='video'
    )
    response = request.execute()
    video_data = []
    for item in response['items']:
        video_id = item['id']['videoId']
        title = item['snippet']['title']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        metadata = {
            'channel': item['snippet'].get('channelTitle', ''),
            'published_at': item['snippet'].get('publishedAt', ''),
            'description': item['snippet'].get('description', ''),
            'topic': query
        }
        video_data.append((title, video_url, metadata))
    logging.debug(f"get_youtube_videos returned {len(video_data)} videos for query '{query}'")
    return video_data

def download_youtube_video(url, download_dir):
    """
    Downloads a YouTube video using yt-dlp and returns the local filename.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            local_filename = ydl.prepare_filename(info)
            logging.info(f"Downloaded video file: {local_filename}")
            return local_filename
    except Exception as e:
        logging.error(f"Error downloading video from {url}: {e}")
        return None

# -------------------------------
# Stage 3: Audio Extraction and Processing
# -------------------------------

def extract_audio(video_file, output_dir):
    """
    Extracts audio from a video file using FFmpeg and returns the local MP3 file path.
    """
    base = os.path.splitext(os.path.basename(video_file))[0]
    audio_path = os.path.join(output_dir, base + '.mp3')
    try:
        ffmpeg.input(video_file).output(audio_path, format='mp3', acodec='libmp3lame').run(quiet=True)
        logging.info(f"Extracted audio: {audio_path}")
        return audio_path
    except Exception as e:
        logging.error(f"Error extracting audio from {video_file}: {e}")
        return None

def clean_audio(input_audio, cleaned_output):
    """Applies noise reduction using librosa and noisereduce and saves as WAV."""
    try:
        y, sr = librosa.load(input_audio, sr=None)
        # Add check for zero values in y
        if np.all(y == 0):
            logging.warning(f"Audio data in {input_audio} is all zeros. Skipping noise reduction.")
            reduced_noise = y  # or handle it differently, like raising an exception
        else:
            reduced_noise = nr.reduce_noise(y=y, sr=sr)

        sf.write(cleaned_output, reduced_noise, sr)
        logging.info(f"Cleaned audio saved to: {cleaned_output}")
    except Exception as e:
        logging.error(f"Error cleaning audio {input_audio}: {e}")
def normalize_audio(input_audio, normalized_output):
    """
    Normalizes audio volume using pydub and exports as MP3.
    """
    try:
        audio = AudioSegment.from_file(input_audio)
        normalized_audio = audio.apply_gain(-audio.dBFS)
        normalized_audio.export(normalized_output, format="mp3")
        logging.info(f"Normalized audio saved to: {normalized_output}")
    except Exception as e:
        logging.error(f"Error normalizing audio {input_audio}: {e}")

def process_audio_file(audio_file):
    """
    Combines cleaning and normalization of a raw audio file.
    Returns the final cleaned and normalized audio filename.
    """
    base = os.path.splitext(os.path.basename(audio_file))[0]
    cleaned_wav = os.path.join(LOCAL_DOWNLOAD_DIR, f"cleaned_{base}.wav")
    normalized_mp3 = os.path.join(LOCAL_DOWNLOAD_DIR, f"final_{base}.mp3")

    clean_audio(audio_file, cleaned_wav)
    normalize_audio(cleaned_wav, normalized_mp3)

    # Cleanup temporary cleaned file
    if os.path.exists(cleaned_wav):
        os.remove(cleaned_wav)
    return normalized_mp3

# -------------------------------
# Stage 4: Transcription and BigQuery Insertion
# -------------------------------

from google.cloud.speech_v1 import SpeechClient, RecognitionConfig, RecognitionAudio
from google.api_core.retry import Retry
import logging

# (re)instantiate your client once
speech_client = SpeechClient()

def transcribe_audio_gcs(gcs_uri: str, use_long_running: bool = False) -> str:
    config = RecognitionConfig(
        encoding=RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="en-US"
    )
    audio = RecognitionAudio(uri=gcs_uri)

    try:
        if use_long_running:
            op = speech_client.long_running_recognize(config=config, audio=audio, retry=speech_retry)
            response = op.result(timeout=300)
        else:
            response = speech_client.recognize(config=config, audio=audio, retry=speech_retry)

        transcript = " ".join(r.alternatives[0].transcript for r in response.results)
        logging.info(f"Transcribed {gcs_uri} → {len(response.results)} segments")
        return transcript

    except Exception:
        logging.exception(f"Transcription error for {gcs_uri}")
        return ""


def ensure_table_columns(dataset_id, table_id, row_data):
    """
    Ensures that all keys in row_data exist as columns in the BigQuery table.
    Adds missing columns as STRING fields.
    """
    try:
        client = bigquery.Client(project=PROJECT_ID)
        table_ref = client.dataset(dataset_id).table(table_id)
        table = client.get_table(table_ref)
        existing_fields = {field.name for field in table.schema}
        new_fields = []
        for key in row_data:
            if key not in existing_fields:
                new_fields.append(bigquery.SchemaField(key, "STRING"))
        if new_fields:
            new_schema = table.schema[:] + new_fields
            table.schema = new_schema
            table = client.update_table(table, ["schema"])
            logging.info(f"Updated table schema with new columns: {[f.name for f in new_fields]}")
    except Exception as e:
        logging.error(f"Error ensuring table columns: {e}")

def save_to_bigquery(dataset_id, table_id, video_title, audio_gcs_path, transcript, metadata):
    """
    Inserts the metadata and transcript into BigQuery after ensuring the table has necessary columns.
    """
    row_data = {
        'video_title': video_title,
        'audio_gcs_path': audio_gcs_path,
        'transcript': transcript,
        'upload_timestamp': datetime.utcnow().isoformat(),
        'channel': metadata.get('channel', ''),
        'published_at': metadata.get('published_at', ''),
        'description': metadata.get('description', ''),
        'topic': metadata.get('topic', '')
    }
    ensure_table_columns(dataset_id, table_id, row_data)

    try:
        client = bigquery.Client(project=PROJECT_ID)
        table_ref = client.dataset(dataset_id).table(table_id)
        errors = client.insert_rows_json(table_ref, [row_data])
        if errors:
            logging.error(f"Error inserting data for {video_title}: {errors}")
        else:
            logging.info(f"Data successfully inserted for {video_title}")
    except Exception as e:
        logging.error(f"Error inserting row into BigQuery for {video_title}: {e}")

# -------------------------------
# Stage 5: Pipeline Stages
# -------------------------------

def ingest_videos():
    """
    Stage A: Video Ingestion Pipeline.
    For each topic, search YouTube, download new videos (avoiding duplicates via GCS),
    extract audio, and upload raw audio to GCS.
    """
    all_videos = []
    for topic in TOPICS:
        logging.info(f"Fetching videos for topic: {topic}")
        videos = get_youtube_videos(API_KEY, topic, MAX_VIDEOS_PER_TOPIC)
        all_videos.extend(videos)
        logging.info(f"Found {len(videos)} videos for topic: {topic}")

    logging.info(f"Total videos fetched: {len(all_videos)}")

    for title, video_url, metadata in all_videos:
        logging.info(f"Processing video: {title} (Topic: {metadata['topic']})")
        try:
            video_local = download_youtube_video(video_url, LOCAL_DOWNLOAD_DIR)
            if video_local is None:
                continue  # Skip if download failed

            video_filename = os.path.basename(video_local)

            # Check for duplicate video in GCS (folder: videos/)
            if file_exists_in_gcs(video_filename, "videos"):
                logging.info(f"Video '{video_filename}' already exists in GCS. Skipping download.")
                os.remove(video_local)
                continue

            # Upload video to GCS under 'videos/' folder
            video_blob = f"videos/{video_filename}"
            upload_to_gcs(video_local, video_blob)

            # Extract audio from the downloaded video
            raw_audio = extract_audio(video_local, LOCAL_DOWNLOAD_DIR)
            if raw_audio is None:
                continue

            audio_filename = os.path.basename(raw_audio)

            # Check for duplicate raw audio in GCS (folder: audio/)
            if not file_exists_in_gcs(audio_filename, "audio"):
                audio_blob = f"audio/{audio_filename}"
                upload_to_gcs(raw_audio, audio_blob)
            else:
                logging.info(f"Raw audio '{audio_filename}' already exists in GCS. Skipping upload.")

            # Cleanup local files
            for f in [video_local, raw_audio]:
                if os.path.exists(f):
                    os.remove(f)
            logging.info(f"Completed ingestion for video: {title}")
        except Exception as e:
            logging.error(f"Error processing video '{title}': {e}")

def process_audios():
    """
    Stage B: Audio Processing Pipeline.
    Processes raw audio files from GCS:
      - Checks for duplicate cleaned audio in 'cleaned_audio/' folder.
      - Downloads raw audio locally and processes it (cleaning and normalization).
      - Uploads cleaned audio to GCS.
      - Transcribes the cleaned audio and inserts corresponding metadata into BigQuery.
    """
    raw_audio_files = list_gcs_files("audio")
    cleaned_audio_files = list_gcs_files("cleaned_audio")
    logging.info(f"Found {len(raw_audio_files)} raw audio files in GCS.")

    for raw_blob in raw_audio_files:
        # Extract base filename
        base = os.path.splitext(os.path.basename(raw_blob))[0]
        cleaned_blob = f"cleaned_audio/final_{base}.mp3"  # Construct cleaned audio blob name

        if cleaned_blob in cleaned_audio_files:
            logging.info(f"Cleaned audio for '{base}' already exists. Skipping processing.")
            continue

        try:
            local_raw = os.path.join(LOCAL_DOWNLOAD_DIR, os.path.basename(raw_blob))
            download_from_gcs(raw_blob, local_raw)
            local_cleaned = process_audio_file(local_raw)
            upload_to_gcs(local_cleaned, cleaned_blob)
            cleaned_gcs_uri = f"gs://{BUCKET_NAME}/{cleaned_blob}"

            # Call transcribe_audio_gcs AFTER uploading the file
            transcript = transcribe_audio_gcs(cleaned_gcs_uri)  # Perform transcription

            metadata = {"topic": "unknown"}  # You may update this with additional metadata if available
            save_to_bigquery(DATASET_ID, TABLE_ID, base, cleaned_gcs_uri, transcript, metadata)

            # Cleanup local files
            for f in [local_raw, local_cleaned]:
                if os.path.exists(f):
                    os.remove(f)
            logging.info(f"Processed and annotated audio: {base}")
        except Exception as e:
            logging.error(f"Error processing audio '{base}': {e}")

def main():
    """
    Main execution function. Runs the two pipeline stages sequentially.
    """
    logging.info("=== Stage A: Video Ingestion ===")
    ingest_videos()
    logging.info("=== Stage B: Audio Processing, Transcription & BigQuery Insertion ===")
    process_audios()
    logging.info("Pipeline completed successfully.")

if __name__ == '__main__':
    main()

import numpy as np, soundfile as sf, librosa
def seg_snr(ref, deg, frame_len=0.02, hop=0.01, sr=16000):
    frames_r = librosa.util.frame(ref, frame_length=int(sr*frame_len), hop_length=int(sr*hop))
    frames_d = librosa.util.frame(deg, frame_length=int(sr*frame_len), hop_length=int(sr*hop))
    snrs = []
    for r, d in zip(frames_r.T, frames_d.T):
        num = np.sum(r**2)
        den = np.sum((r-d)**2) + 1e-8
        snrs.append(10*np.log10(num/den))
    return np.mean(snrs)

"""visualize the effects of audio cleaning by comparing the spectrograms of the raw and cleaned audio files."""

import random
import matplotlib.pyplot as plt
    # === Spectrogram Comparison Inline ===
print("\n=== Generating random spectrogram comparison ===")

    # 1. List all raw-audio blobs in GCS
raw_blobs = list_gcs_files("audio")
if not raw_blobs:
        print("No raw audio found in GCS — skipping spectrogram.")
else:
        # 2. Pick one at random
        raw_blob = random.choice(list(raw_blobs))
        base     = os.path.splitext(os.path.basename(raw_blob))[0]
        cleaned_blob = f"cleaned_audio/final_{base}.mp3"

        # 3. Ensure the cleaned version exists
        cleaned_list = list_gcs_files("cleaned_audio")
        if cleaned_blob not in cleaned_list:
            print(f"No cleaned audio for '{base}' — skipping spectrogram.")
        else:
            # 4. Download raw and cleaned locally
            local_raw   = os.path.join(LOCAL_DOWNLOAD_DIR,   f"{base}.mp3")
            local_clean = os.path.join(LOCAL_DOWNLOAD_DIR, f"final_{base}.mp3")
            for blob_name, lp in [(raw_blob, local_raw), (cleaned_blob, local_clean)]:
                client_storage.bucket(BUCKET_NAME).blob(blob_name).download_to_filename(lp)
                print(f"Downloaded {blob_name} → {lp}")

            # 5. Load and compute STFT spectrograms
            y_raw, sr     = librosa.load(local_raw,   sr=None)
            y_clean, _    = librosa.load(local_clean, sr=None)
            S_raw         = np.abs(librosa.stft(y_raw))
            S_clean       = np.abs(librosa.stft(y_clean))
            DB_raw        = librosa.amplitude_to_db(S_raw,   ref=np.max)
            DB_clean      = librosa.amplitude_to_db(S_clean, ref=np.max)

            # 6. Plot side-by-side
            plt.figure(figsize=(14,5))
            plt.subplot(1,2,1)
            librosa.display.specshow(DB_raw,   sr=sr, y_axis='log', x_axis='time')
            plt.title('Raw Audio')
            plt.colorbar(format='%+2.0f dB')

            plt.subplot(1,2,2)
            librosa.display.specshow(DB_clean, sr=sr, y_axis='log', x_axis='time')
            plt.title('Cleaned Audio')
            plt.colorbar(format='%+2.0f dB')

            plt.tight_layout()

            # 7. Save figure
            fig_path = "figures/spectrogram_pipeline.png"
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            plt.savefig(fig_path, dpi=300)
            plt.close()
            print(f"Spectrogram saved to {fig_path}")

            # 8. Cleanup local temp files
            for fp in (local_raw, local_clean):
                try: os.remove(fp)
                except OSError: pass

"""4.6.1 Objective Metrics: Global SNR over 10 Random Pairs"""

import os, random, subprocess, tempfile
import numpy as np
import soundfile as sf
from google.cloud import storage

# Config
PROJECT_ID   = os.environ.get("GCP_PROJECT_ID", "")
BUCKET_NAME  = "youtube-videos-extracted"
RAW_PREFIX   = "audio/"
CLEAN_PREFIX = "cleaned_audio/"
SR           = 16000
N_SAMPLES    = 10

# Initialize GCS
client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

# Helper: convert MP3→WAV mono 16 kHz
def to_wav(src, dst):
    subprocess.run([
        "ffmpeg","-y","-i",src,
        "-ac","1","-ar",str(SR),
        "-sample_fmt","s16",dst
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Helper: compute SNR between two WAVs
def compute_snr(raw_wav, clean_wav):
    y_raw, _   = sf.read(raw_wav)
    y_clean, _ = sf.read(clean_wav)

    # Convert stereo to mono if necessary:
    if y_raw.ndim > 1:  # Check if y_raw is stereo
        y_raw = y_raw.mean(axis=1)  # Average channels to get mono
    if y_clean.ndim > 1:  # Check if y_clean is stereo
        y_clean = y_clean.mean(axis=1)  # Average channels to get mono

    L = min(len(y_raw), len(y_clean))
    y_raw, y_clean = y_raw[:L], y_clean[:L]
    return 10 * np.log10(
        np.sum(y_clean**2) /
        (np.sum((y_raw - y_clean)**2) + 1e-12)
    )

snr_list = []

# Main loop: sample N_SAMPLES pairs
raw_blobs = [b.name for b in bucket.list_blobs(prefix=RAW_PREFIX) if b.name.endswith(".mp3")]
for _ in range(N_SAMPLES):
    raw_blob   = random.choice(raw_blobs)
    base       = os.path.basename(raw_blob)
    clean_blob = f"{CLEAN_PREFIX}final_{base}"

    # Download to temp
    tmp = tempfile.mkdtemp(prefix="snr_")
    raw_mp3   = os.path.join(tmp, base)
    clean_mp3 = os.path.join(tmp, f"final_{base}")
    bucket.blob(raw_blob).download_to_filename(raw_mp3)
    bucket.blob(clean_blob).download_to_filename(clean_mp3)

    # Convert to WAV
    raw_wav   = os.path.join(tmp, "raw.wav")
    clean_wav = os.path.join(tmp, "clean.wav")
    to_wav(raw_mp3, raw_wav)
    to_wav(clean_mp3, clean_wav)

    # Compute and record SNR
    snr_val = compute_snr(raw_wav, clean_wav)
    snr_list.append(snr_val)

    # Cleanup
    import shutil
    shutil.rmtree(tmp)

# Results
avg_snr = np.mean(snr_list)
print("Sample\tSNR (dB)")
for i, v in enumerate(snr_list, 1):
    print(f"{i}\t{v:.2f}")
print(f"\nAverage SNR improvement over {N_SAMPLES} samples: {avg_snr:.2f} dB")

"""4.6.2 Segmental SNR"""

import os
import random
import tempfile
import subprocess
import numpy as np
import soundfile as sf
from google.cloud import storage
from tabulate import tabulate

# ── CONFIG ───────────────────────────────────────────────────────
PROJECT_ID   = os.environ.get("GCP_PROJECT_ID", "")
BUCKET_NAME  = "youtube-videos-extracted"
RAW_PREFIX   = "audio/"
CLEAN_PREFIX = "cleaned_audio/"
SR           = 16000       # target sample rate
N_SAMPLES    = 10          # how many random files to test
FRAME_MS     = 30
HOP_MS       = 15

# ── GCS CLIENT ──────────────────────────────────────────────────
client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

# ── HELPERS ────────────────────────────────────────────────────
def ffmpeg_to_wav(src_mp3: str, dst_wav: str, sr: int = SR):
    """
    Uses ffmpeg to convert any audio file to mono wav @ sr Hz.
    """
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", src_mp3,
            "-ac", "1",
            "-ar", str(sr),
            "-vn",
            dst_wav
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def segmental_snr(y_ref, y_deg, sr=SR, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """Compute segmental SNR (in dB) between two mono numpy arrays."""
    L = min(len(y_ref), len(y_deg))
    y_ref, y_deg = y_ref[:L], y_deg[:L]

    frame_len = int(frame_ms * sr / 1000)
    hop_len   = int(hop_ms   * sr / 1000)
    vals = []
    for start in range(0, L - frame_len + 1, hop_len):
        seg_r = y_ref[start:start + frame_len]
        seg_c = y_deg[start:start + frame_len]
        if np.sum(seg_r**2) < 1e-8:
            continue
        noise = seg_r - seg_c
        vals.append(10 * np.log10((np.sum(seg_r**2) + 1e-12) /
                                  (np.sum(noise**2) + 1e-12)))
    return float(np.mean(vals)) if vals else None

# ── MAIN EVAL ──────────────────────────────────────────────────
def eval_segmental_snr(n=N_SAMPLES):
    # list all raw audio blobs
    all_raw = [b.name for b in bucket.list_blobs(prefix=RAW_PREFIX) if b.name.endswith(".mp3")]
    sample = random.sample(all_raw, min(n, len(all_raw)))
    results = []

    for raw_blob in sample:
        title      = os.path.basename(raw_blob).rsplit(".", 1)[0]
        clean_blob = f"{CLEAN_PREFIX}final_{title}.mp3"

        if not bucket.blob(clean_blob).exists():
            print(f"[skip] no cleaned for {title}")
            continue

        with tempfile.TemporaryDirectory() as td:
            raw_mp3   = os.path.join(td, "raw.mp3")
            clean_mp3 = os.path.join(td, "clean.mp3")
            wav_raw   = os.path.join(td, "raw.wav")
            wav_clean = os.path.join(td, "clean.wav")

            # download
            bucket.blob(raw_blob).download_to_filename(raw_mp3)
            bucket.blob(clean_blob).download_to_filename(clean_mp3)

            # ffmpeg → WAV@16k,mono
            ffmpeg_to_wav(raw_mp3, wav_raw,   sr=SR)
            ffmpeg_to_wav(clean_mp3, wav_clean, sr=SR)

            # read
            y_raw,   _ = sf.read(wav_raw)
            y_clean, _ = sf.read(wav_clean)

            # compute
            seg_val = segmental_snr(y_raw, y_clean, sr=SR)
            results.append((title, seg_val))

    # print table
    print("\nSegmental SNR per sample (dB):")
    print(tabulate(results, headers=["Title","Seg. SNR (dB)"], floatfmt=".2f"))
    if results:
        avg = np.mean([r[1] for r in results])
        print(f"\n→ Average (over {len(results)}): {avg:.2f} dB")

# ── ENTRY POINT ────────────────────────────────────────────────
if __name__ == "__main__":
    eval_segmental_snr()

import os
import random
import tempfile
import subprocess
import numpy as np
import soundfile as sf
from google.cloud import storage
from tabulate import tabulate

# ── CONFIG ───────────────────────────────────────────────────────
PROJECT_ID   = os.environ.get("GCP_PROJECT_ID", "")
BUCKET_NAME  = "youtube-videos-extracted"
RAW_PREFIX   = "audio/"
CLEAN_PREFIX = "cleaned_audio/"
SR           = 16000       # target sample rate
N_SAMPLES    = 10          # random files to test
FRAME_MS     = 30
HOP_MS       = 15

# ── GCS CLIENT ──────────────────────────────────────────────────
client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

# ── HELPERS ────────────────────────────────────────────────────
def ffmpeg_to_wav(src_mp3: str, dst_wav: str, sr: int = SR):
    """Convert any audio file to mono WAV @ sr Hz via ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_mp3, "-ac", "1", "-ar", str(sr), "-vn", dst_wav],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

def compute_global_snr(y_ref, y_deg):
    """
    Global SNR = 10 log10( sum(ref^2) / sum((ref-deg)^2) )
    """
    L = min(len(y_ref), len(y_deg))
    r, d = y_ref[:L], y_deg[:L]
    noise = r - d
    return 10 * np.log10((np.sum(r**2) + 1e-12) / (np.sum(noise**2) + 1e-12))

def segmental_snr(y_ref, y_deg, sr=SR, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """Compute segmental SNR between two mono arrays."""
    L = min(len(y_ref), len(y_deg))
    y_ref, y_deg = y_ref[:L], y_deg[:L]
    fl = int(frame_ms * sr / 1000)
    hl = int(hop_ms   * sr / 1000)
    vals = []
    for start in range(0, L - fl + 1, hl):
        seg_r = y_ref[start:start+fl]
        seg_c = y_deg[start:start+fl]
        if np.sum(seg_r**2) < 1e-8:
            continue
        noise = seg_r - seg_c
        vals.append(10 * np.log10((np.sum(seg_r**2) + 1e-12) /
                                  (np.sum(noise**2) + 1e-12)))
    return float(np.mean(vals)) if vals else None

# ── MAIN EVAL ──────────────────────────────────────────────────
def eval_snr_pairs(n=N_SAMPLES):
    all_raw = [b.name for b in bucket.list_blobs(prefix=RAW_PREFIX) if b.name.endswith(".mp3")]
    sample = random.sample(all_raw, min(n, len(all_raw)))
    table = []

    for raw_blob in sample:
        title      = os.path.basename(raw_blob).rsplit(".",1)[0]
        clean_blob = f"{CLEAN_PREFIX}final_{title}.mp3"
        if not bucket.blob(clean_blob).exists():
            print(f"[skip] no cleaned for {title}")
            continue

        with tempfile.TemporaryDirectory() as td:
            raw_mp3   = os.path.join(td, "raw.mp3")
            clean_mp3 = os.path.join(td, "clean.mp3")
            wav_raw   = os.path.join(td, "raw.wav")
            wav_clean = os.path.join(td, "clean.wav")

            # download
            bucket.blob(raw_blob).download_to_filename(raw_mp3)
            bucket.blob(clean_blob).download_to_filename(clean_mp3)

            # to WAV@16k,mono
            ffmpeg_to_wav(raw_mp3, wav_raw,   sr=SR)
            ffmpeg_to_wav(clean_mp3, wav_clean, sr=SR)

            # read
            y_raw,   _ = sf.read(wav_raw)
            y_clean, _ = sf.read(wav_clean)

            # compute both metrics
            g_snr = compute_global_snr(y_raw, y_clean)
            s_snr = segmental_snr(y_raw, y_clean, sr=SR)

            table.append((title, g_snr, s_snr))

    # display
    print("\nSNR Results:")
    print(tabulate(table, headers=["Title","Global SNR (dB)","Segmental SNR (dB)"],
                   floatfmt=".2f", showindex=True))
    if table:
        avg_g = np.mean([r[1] for r in table])
        avg_s = np.mean([r[2] for r in table if r[2] is not None])
        print(f"\n→ Avg Global SNR: {avg_g:.2f} dB")
        print(f"→ Avg Segmental SNR: {avg_s:.2f} dB")

# ── ENTRY POINT ────────────────────────────────────────────────
if __name__ == "__main__":
    eval_snr_pairs()

"""4.6.3 Mel-Cepstral Distortion (MCD) (GCS-backed)"""

import os, random, subprocess, tempfile
import numpy as np
import librosa
from google.cloud import storage

# Configuration (reuse from above)
# PROJECT_ID, BUCKET_NAME, RAW_PREFIX, CLEAN_PREFIX, SR

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

def download_and_convert(blob_name):
    tmp = tempfile.mkdtemp(prefix="mcd_")
    mp3_path = os.path.join(tmp, os.path.basename(blob_name))
    wav_path = os.path.join(tmp, "out.wav")
    bucket.blob(blob_name).download_to_filename(mp3_path)
    subprocess.run([
        "ffmpeg","-y","-i",mp3_path,
        "-ac","1","-ar",str(SR),
        "-sample_fmt","s16",wav_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp, wav_path

def mel_cepstral_distortion(raw_wav, clean_wav, n_mfcc=13):
    y_r, _ = librosa.load(raw_wav, sr=SR, mono=True)
    y_c, _ = librosa.load(clean_wav, sr=SR, mono=True)
    mfcc_r = librosa.feature.mfcc(y=y_r, sr=SR, n_mfcc=n_mfcc)
    mfcc_c = librosa.feature.mfcc(y=y_c, sr=SR, n_mfcc=n_mfcc)
    T = min(mfcc_r.shape[1], mfcc_c.shape[1])
    mfcc_r, mfcc_c = mfcc_r[:,:T], mfcc_c[:,:T]
    coef = (10.0/np.log(10.0))*np.sqrt(2.0)
    diff = mfcc_r - mfcc_c
    per_frame = np.linalg.norm(diff, axis=0)
    return coef * np.mean(per_frame)

# Random selection
raw_list = [b.name for b in bucket.list_blobs(prefix=RAW_PREFIX) if b.name.endswith(".mp3")]
raw_blob = random.choice(raw_list)
clean_blob = f"{CLEAN_PREFIX}final_{os.path.basename(raw_blob)}"

# Download & convert
tmp1, raw_wav   = download_and_convert(raw_blob)
tmp2, clean_wav = download_and_convert(clean_blob)

# Compute MCD
mcd = mel_cepstral_distortion(raw_wav, clean_wav)
print(f"Mel-Cepstral Distortion for '{os.path.basename(raw_blob)}': {mcd:.2f} dB")

# Cleanup
import shutil
shutil.rmtree(tmp1)
shutil.rmtree(tmp2)