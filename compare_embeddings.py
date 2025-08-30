#!/usr/bin/env python3
"""
Compare embeddings to verify similarity patterns.
"""
import numpy as np
import apsw
from pathlib import Path
from io import BytesIO
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load_embeddings_from_db(db_path):
    """Load all embeddings from the database."""
    conn = apsw.Connection(str(db_path))
    cursor = conn.cursor()
    
    embeddings_dict = {}
    for song_id, embedding_blob in cursor.execute("SELECT song_id, embedding FROM songs_vectors"):
        # Load the numpy array from the blob
        embedding_io = BytesIO(embedding_blob)
        embedding = np.load(embedding_io)
        embeddings_dict[song_id] = embedding
    
    return embeddings_dict

def calculate_similarity_matrix(embeddings_dict, distance_metric='cosine'):
    """Calculate pairwise similarity matrix."""
    song_ids = list(embeddings_dict.keys())
    n = len(song_ids)
    
    # Create distance matrix
    distance_matrix = np.zeros((n, n))
    
    for i, id1 in enumerate(song_ids):
        for j, id2 in enumerate(song_ids):
            if distance_metric == 'cosine':
                distance_matrix[i, j] = cosine(embeddings_dict[id1], embeddings_dict[id2])
            elif distance_metric == 'euclidean':
                distance_matrix[i, j] = euclidean(embeddings_dict[id1], embeddings_dict[id2])
    
    # Convert distance to similarity (for cosine: similarity = 1 - distance)
    if distance_metric == 'cosine':
        similarity_matrix = 1 - distance_matrix
    else:
        # For euclidean, convert to similarity using exponential decay
        similarity_matrix = np.exp(-distance_matrix / np.max(distance_matrix))
    
    return similarity_matrix, song_ids

def analyze_similarities(embeddings_dict):
    """Analyze and print similarity patterns."""
    print("\n=== EMBEDDING ANALYSIS ===")
    print(f"Generated embeddings for {len(embeddings_dict)} audio files:")
    for song_id, embedding in embeddings_dict.items():
        print(f"  - {song_id}: {embedding.shape}")
    
    if len(embeddings_dict) < 2:
        print("Need at least 2 embeddings for comparison")
        return
    
    # Calculate similarity matrices
    cosine_sim, song_ids = calculate_similarity_matrix(embeddings_dict, 'cosine')
    euclidean_sim, _ = calculate_similarity_matrix(embeddings_dict, 'euclidean')
    
    print(f"\n=== SIMILARITY ANALYSIS ===")
    
    # Expected groupings based on audio characteristics:
    # Similar: major chords (c_major, g_major, f_major)
    # Similar: noise sounds (white_noise, pink_noise)
    # Different: bass vs bell vs complex vs noise
    
    chord_files = ['c_major_chord', 'g_major_chord', 'f_major_chord']
    noise_files = ['white_noise', 'pink_noise']
    tonal_files = ['bass_sound', 'bell_sound', 'complex_harmonic']
    
    def get_avg_similarity(files, sim_matrix, song_ids, metric_name):
        """Calculate average similarity within a group."""
        indices = [song_ids.index(f) for f in files if f in song_ids]
        if len(indices) < 2:
            return None
        
        similarities = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                similarities.append(sim_matrix[indices[i], indices[j]])
        
        avg_sim = np.mean(similarities)
        print(f"  {metric_name} - Average similarity within {files}: {avg_sim:.3f}")
        return avg_sim
    
    def get_cross_group_similarity(group1, group2, sim_matrix, song_ids, metric_name):
        """Calculate average similarity between groups."""
        indices1 = [song_ids.index(f) for f in group1 if f in song_ids]
        indices2 = [song_ids.index(f) for f in group2 if f in song_ids]
        
        if len(indices1) == 0 or len(indices2) == 0:
            return None
        
        similarities = []
        for i in indices1:
            for j in indices2:
                similarities.append(sim_matrix[i, j])
        
        avg_sim = np.mean(similarities)
        print(f"  {metric_name} - Average similarity between {group1} and {group2}: {avg_sim:.3f}")
        return avg_sim
    
    print("\nCOSINE SIMILARITY:")
    chord_cosine = get_avg_similarity(chord_files, cosine_sim, song_ids, "Cosine")
    noise_cosine = get_avg_similarity(noise_files, cosine_sim, song_ids, "Cosine")
    cross_cosine = get_cross_group_similarity(chord_files, noise_files, cosine_sim, song_ids, "Cosine")
    
    print("\nEUCLIDEAN SIMILARITY:")
    chord_euclidean = get_avg_similarity(chord_files, euclidean_sim, song_ids, "Euclidean")
    noise_euclidean = get_avg_similarity(noise_files, euclidean_sim, song_ids, "Euclidean")
    cross_euclidean = get_cross_group_similarity(chord_files, noise_files, euclidean_sim, song_ids, "Euclidean")
    
    # Print detailed similarity matrix
    print(f"\n=== DETAILED COSINE SIMILARITY MATRIX ===")
    print("Files:", song_ids)
    print("Similarity matrix (1.0 = identical, 0.0 = completely different):")
    for i, id1 in enumerate(song_ids):
        print(f"{id1:15s}", end=" ")
        for j, id2 in enumerate(song_ids):
            print(f"{cosine_sim[i,j]:.3f}", end=" ")
        print()
    
    # Find most and least similar pairs
    print(f"\n=== MOST AND LEAST SIMILAR PAIRS ===")
    n = len(song_ids)
    similarities = []
    
    for i in range(n):
        for j in range(i+1, n):
            similarities.append((cosine_sim[i,j], song_ids[i], song_ids[j]))
    
    similarities.sort(reverse=True)
    
    print("MOST SIMILAR pairs (cosine similarity):")
    for sim, id1, id2 in similarities[:3]:
        print(f"  {id1} <-> {id2}: {sim:.3f}")
    
    print("LEAST SIMILAR pairs (cosine similarity):")
    for sim, id1, id2 in similarities[-3:]:
        print(f"  {id1} <-> {id2}: {sim:.3f}")
    
    return cosine_sim, euclidean_sim, song_ids

def visualize_embeddings(embeddings_dict, output_path="embedding_visualization.png"):
    """Create a 2D visualization of the embeddings."""
    if len(embeddings_dict) < 3:
        print("Need at least 3 embeddings for visualization")
        return
    
    # Prepare data
    song_ids = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[sid] for sid in song_ids])
    
    # Reduce dimensions using PCA and t-SNE
    pca_components = min(min(50, embeddings.shape[1]), embeddings.shape[0] - 1)
    pca = PCA(n_components=pca_components)
    embeddings_pca = pca.fit_transform(embeddings)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(3, len(song_ids)-1))
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Color code by type
    colors = []
    labels = []
    for sid in song_ids:
        if 'chord' in sid:
            colors.append('blue')
            labels.append('Chord')
        elif 'noise' in sid:
            colors.append('red')
            labels.append('Noise')
        elif 'bass' in sid:
            colors.append('green')
            labels.append('Bass')
        elif 'bell' in sid:
            colors.append('orange')
            labels.append('Bell')
        else:
            colors.append('purple')
            labels.append('Other')
    
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=100, alpha=0.7)
    
    # Add labels
    for i, sid in enumerate(song_ids):
        plt.annotate(sid.replace('_', '\n'), (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('Audio Embedding Visualization (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    
    # Create legend
    unique_labels = list(set(labels))
    unique_colors = ['blue', 'red', 'green', 'orange', 'purple'][:len(unique_labels)]
    for label, color in zip(unique_labels, unique_colors):
        plt.scatter([], [], c=color, label=label, s=100, alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved to {output_path}")

def main():
    db_path = Path("metadata/songs.db")
    
    if not db_path.exists():
        print("Database not found. Please run the embedding generation first.")
        return
    
    # Load embeddings
    print("Loading embeddings from database...")
    embeddings_dict = load_embeddings_from_db(db_path)
    
    if not embeddings_dict:
        print("No embeddings found in database. Please run the embedding generation first.")
        return
    
    # Analyze similarities
    cosine_sim, euclidean_sim, song_ids = analyze_similarities(embeddings_dict)
    
    # Create visualization
    visualize_embeddings(embeddings_dict)
    
    print(f"\n=== CONCLUSIONS ===")
    print("Expected results:")
    print("- Chord files (c_major, g_major, f_major) should be similar to each other")
    print("- Noise files (white_noise, pink_noise) should be similar to each other")  
    print("- Chord files should be different from noise files")
    print("- Bass, bell, and complex harmonic should be distinct from each other")
    print("\nIf the embeddings show these patterns, the system is working correctly!")

if __name__ == "__main__":
    main()