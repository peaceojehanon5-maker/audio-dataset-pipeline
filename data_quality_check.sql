-- Data-quality check on the BigQuery table that pipeline.py writes to.
-- From Appendix B of the dissertation report (May 2025).
-- Replace YOUR_PROJECT_ID with your Google Cloud project ID before running.

WITH counts AS (
  SELECT
    COUNT(*) AS total_rows,
    COUNTIF(video_title IS NULL) AS null_video_title,
    COUNTIF(audio_gcs_path IS NULL) AS null_audio_gcs_path,
    COUNTIF(transcript IS NULL) AS null_transcript,
    COUNTIF(upload_timestamp IS NULL) AS null_upload_timestamp
  FROM YOUR_PROJECT_ID.my_dataset.audio_metadata
),
transcript_stats AS (
  SELECT
    AVG(ARRAY_LENGTH(SPLIT(transcript, ' '))) AS avg_transcript_length_words,
    MIN(ARRAY_LENGTH(SPLIT(transcript, ' '))) AS min_transcript_length_words,
    MAX(ARRAY_LENGTH(SPLIT(transcript, ' '))) AS max_transcript_length_words
  FROM YOUR_PROJECT_ID.my_dataset.audio_metadata
),
most_common_topic AS (
  SELECT
    topic AS most_common_topic
  FROM YOUR_PROJECT_ID.my_dataset.audio_metadata
  GROUP BY topic
  ORDER BY COUNT(*) DESC
  LIMIT 1
)
SELECT
  counts.total_rows,
  counts.null_video_title,
  counts.null_audio_gcs_path,
  counts.null_transcript,
  counts.null_upload_timestamp,
  transcript_stats.avg_transcript_length_words,
  transcript_stats.min_transcript_length_words,
  transcript_stats.max_transcript_length_words,
  most_common_topic.most_common_topic,
  -- Placeholder: File existence check requires external processing (e.g., Python + GCS API)
  NULL AS file_exists_flag
FROM counts
CROSS JOIN transcript_stats
CROSS JOIN most_common_topic;
