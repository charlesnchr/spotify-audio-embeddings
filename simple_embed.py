#!/usr/bin/env python3
"""
Simple embedding generation script for testing purposes.
"""
import openl3
import soundfile as sf
import numpy as np
import skimage.measure
from pathlib import Path
import sqlite3
from contextlib import closing
import apsw
from io import BytesIO

def max_pool(embeddings):
    """The authors use max-pooling in their paper, so we'll do the same to create a single embedding"""
    return skimage.measure.block_reduce(embeddings, (embeddings.shape[0], 1), np.max)[0]

def process_audio_file(audio_path, model, conn):
    """Process a single audio file and store its embedding."""
    print(f"Processing {audio_path.name}...")

    try:
        # Read the audio file
        audio, sr = sf.read(audio_path)

        # Get embedding using OpenL3
        embeddings, _ = openl3.get_audio_embedding([audio], [sr], batch_size=1, model=model)

        # Max pool to get a single vector per file
        max_pooled = max_pool(embeddings[0])

        # Store in database
        with closing(conn.cursor()) as cursor:
            with conn:
                outfile = BytesIO()
                np.save(outfile, max_pooled.astype(np.double), allow_pickle=False)
                cursor.execute("INSERT OR REPLACE INTO songs_vectors VALUES (?, ?);",
                             (audio_path.stem, outfile.getvalue()))

        print(f"  ✓ Generated {max_pooled.shape} embedding for {audio_path.name}")
        return max_pooled

    except Exception as e:
        print(f"  ✗ Error processing {audio_path.name}: {e}")
        return None

def main():
    # Initialize
    EMBEDDING_SIZE = 512
    print("Loading OpenL3 model...")
    model = openl3.models.load_audio_embedding_model(
        input_repr="mel256",
        content_type="music",
        embedding_size=EMBEDDING_SIZE
    )

    # Connect to database
    db_path = Path("metadata/songs.db")
    conn = apsw.Connection(str(db_path))

    # Get audio files
    previews_dir = Path("previews")
    audio_paths = list(previews_dir.glob("*.mp3"))

    print(f"Found {len(audio_paths)} audio files to process")

    embeddings_dict = {}

    # Process each file
    for audio_path in sorted(audio_paths):
        embedding = process_audio_file(audio_path, model, conn)
        if embedding is not None:
            embeddings_dict[audio_path.stem] = embedding

    print(f"\nSuccessfully processed {len(embeddings_dict)} files")
    return embeddings_dict

if __name__ == "__main__":
    embeddings = main()
