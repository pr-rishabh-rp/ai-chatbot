"""
Create voice embeddings from WAV files for XTTS voice cloning.
This generates the same format as male.json and female.json.
"""

import torch
import json
import os
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def create_voice_embeddings(wav_file_path, output_json_path=None):
    """
    Convert a WAV file to XTTS voice embeddings (gpt_cond_latent and speaker_embedding).
    
    Args:
        wav_file_path: Path to the WAV audio file (10-15 seconds of clear speech)
        output_json_path: Optional path to save JSON. If None, uses wav filename with .json extension
    
    Returns:
        Dictionary containing the embeddings
    """
    
    print(f"Loading XTTS model...")
    
    # Load the XTTS model (same one used by RealtimeTTS)
    config = XttsConfig()
    config.load_json(os.path.join("models", "v2.0.2", "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=os.path.join("models", "v2.0.2"),
        use_deepspeed=False
    )
    
    if torch.cuda.is_available():
        model.cuda()
        print("Using GPU for embedding generation")
    else:
        print("Using CPU for embedding generation")
    
    print(f"Processing: {wav_file_path}")
    
    # Compute speaker latents from the audio file
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[wav_file_path]
    )
    
    # Convert to lists for JSON serialization
    embeddings = {
        "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().tolist(),
        "speaker_embedding": speaker_embedding.cpu().squeeze().tolist()
    }
    
    # Save to JSON
    if output_json_path is None:
        output_json_path = os.path.splitext(wav_file_path)[0] + ".json"
    
    with open(output_json_path, 'w') as f:
        json.dump(embeddings, f)
    
    print(f"Embeddings saved to: {output_json_path}")
    print(f"  - gpt_cond_latent shape: {len(embeddings['gpt_cond_latent'])} elements")
    print(f"  - speaker_embedding shape: {len(embeddings['speaker_embedding'])} elements")
    
    return embeddings


def create_multi_voice_embeddings(wav_files, output_json_path):
    """
    Create embeddings for multiple WAV files and save as a collection (like male.json format).
    
    Args:
        wav_files: List of WAV file paths
        output_json_path: Path to save the combined JSON file
    """
    
    print(f"Loading XTTS model...")
    
    config = XttsConfig()
    config.load_json(os.path.join("models", "v2.0.2", "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=os.path.join("models", "v2.0.2"),
        use_deepspeed=False
    )
    
    if torch.cuda.is_available():
        model.cuda()
        print("Using GPU for embedding generation")
    else:
        print("Using CPU for embedding generation")
    
    all_gpt_cond_latents = []
    speaker_embedding = None  # Same for all voices from same speaker
    
    for i, wav_file in enumerate(wav_files, 1):
        print(f"\nProcessing voice {i}/{len(wav_files)}: {wav_file}")
        
        gpt_cond_latent, spk_emb = model.get_conditioning_latents(
            audio_path=[wav_file]
        )
        
        all_gpt_cond_latents.append(gpt_cond_latent.cpu().squeeze().tolist())
        
        if speaker_embedding is None:
            speaker_embedding = spk_emb.cpu().squeeze().tolist()
    
    # Create the same format as male.json/female.json
    embeddings = {
        "gpt_cond_latent": all_gpt_cond_latents,
        "speaker_embedding": speaker_embedding
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(embeddings, f, indent=2)
    
    print(f"\n✓ All embeddings saved to: {output_json_path}")
    print(f"  - {len(all_gpt_cond_latents)} voice variations")
    print(f"  - gpt_cond_latent shape: {len(all_gpt_cond_latents[0])} elements each")
    print(f"  - speaker_embedding shape: {len(speaker_embedding)} elements")


if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Voice Embedding Generator for XTTS")
    print("=" * 70)
    print()
    
    if len(sys.argv) > 1:
        # Command line mode
        wav_files = sys.argv[1:]
        
        if len(wav_files) == 1:
            # Single file mode
            create_voice_embeddings(wav_files[0])
        else:
            # Multiple files mode
            output_name = input("Enter output filename (e.g., my_voices.json): ").strip()
            if not output_name:
                output_name = "custom_voices.json"
            create_multi_voice_embeddings(wav_files, output_name)
    else:
        # Interactive mode
        print("Usage Options:")
        print()
        print("1. Single WAV file:")
        print("   python create_voice_embeddings.py voice.wav")
        print()
        print("2. Multiple WAV files (creates collection like male.json):")
        print("   python create_voice_embeddings.py voice1.wav voice2.wav voice3.wav")
        print()
        print("=" * 70)
        print()
        
        mode = input("Create embeddings for [1] single file or [2] multiple files? (1/2): ").strip()
        
        if mode == "1":
            wav_file = input("Enter WAV file path: ").strip()
            if os.path.exists(wav_file):
                create_voice_embeddings(wav_file)
            else:
                print(f"Error: File not found: {wav_file}")
        
        elif mode == "2":
            print("\nEnter WAV file paths (one per line, empty line to finish):")
            wav_files = []
            while True:
                path = input(f"  File {len(wav_files)+1}: ").strip()
                if not path:
                    break
                if os.path.exists(path):
                    wav_files.append(path)
                else:
                    print(f"  Warning: File not found: {path}, skipping...")
            
            if wav_files:
                output_name = input("\nEnter output filename (e.g., my_voices.json): ").strip()
                if not output_name:
                    output_name = "custom_voices.json"
                create_multi_voice_embeddings(wav_files, output_name)
            else:
                print("No valid files provided.")
        
        else:
            print("Invalid option.")
