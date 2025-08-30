# Audio Embeddings with OpenL3

Generate deep audio embeddings using OpenL3 for music similarity analysis and clustering.

## Demo
Original demo: https://www.youtube.com/watch?v=bk7PKtHLudE

## Overview

This project uses OpenL3 deep learning models to convert audio into high-dimensional embeddings that capture musical characteristics. These embeddings can then be used for:
- Music similarity analysis
- Genre clustering and visualization  
- Content-based music recommendation
- Audio content analysis

## Technology Stack

- **OpenL3**: Deep audio embedding model (512-dimensional vectors)
- **TensorFlow**: Deep learning framework
- **SciKit-Learn**: Clustering and dimensionality reduction
- **APSW**: SQLite database for embedding storage
- **Matplotlib**: Visualization and plotting
- **SoundFile**: Audio file I/O

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install tensorflow==2.13.0 openl3==0.4.2 soundfile matplotlib numpy scikit-image apsw
   ```

2. **Generate test audio:**
   ```bash
   python create_test_audio.py
   ```

3. **Generate embeddings:**
   ```bash
   python simple_embed.py
   ```

4. **Analyze results:**
   ```bash
   python compare_embeddings.py
   python results_summary.py
   ```

## Performance

- **Processing speed**: ~0.5 seconds per second of audio (CPU)
- **Embedding dimensions**: 512
- **Storage**: ~4KB per 30-second track
- **Model**: OpenL3 (mel256, music content type)


## Files

- `embed.py` - Original batch processing script
- `simple_embed.py` - Simplified single-file processing
- `compare_embeddings.py` - Similarity analysis and visualization
- `create_test_audio.py` - Generate synthetic test audio
- `results_summary.py` - Verification report generator

## Database

Embeddings are stored in `metadata/songs.db` (SQLite):
- **Table**: `songs_vectors`
- **Columns**: `song_id` (TEXT), `embedding` (BLOB)
- **Format**: 512-dimensional float64 vectors

## Original Project Notes

The original system processed 945 songs from Spotify's 126 genres, with plans to scale to 100K songs. Key insights:
- Rock music clusters near other rock, Spanish music near other Spanish music
- Max pooling works better than mean pooling for combining time-series embeddings
- Interactive visualization with audio playback provides intuitive music exploration
- Clustering quality improves with more data points relative to cluster count
