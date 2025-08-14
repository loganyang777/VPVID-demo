#!/usr/bin/env python3
"""
Generate spectrograms for all audio files in the input_wsj0c3 directory.
This script creates 360x200 pixel spectrograms for web display.

Requirements:
    pip install librosa matplotlib soundfile
"""

import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import soundfile as sf

def create_spectrogram(audio_path, output_path, figsize=(5, 2.78), dpi=72):
    """
    Create a spectrogram from an audio file.
    
    Args:
        audio_path: Path to the input audio file
        output_path: Path to save the spectrogram image
        figsize: Figure size in inches (width, height)
        dpi: DPI for output image (72 DPI * 5 inches = 360px width, 72 DPI * 2.78 inches ≈ 200px height)
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Compute mel-spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Create figure with specific size
        plt.figure(figsize=figsize, dpi=dpi)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        # Plot spectrogram
        librosa.display.specshow(S_dB, 
                                sr=sr, 
                                x_axis='time', 
                                y_axis='mel',
                                fmax=8000,
                                cmap='viridis')
        
        # Remove axes and labels for clean web display
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save with high quality
        plt.savefig(output_path, 
                   bbox_inches='tight', 
                   pad_inches=0, 
                   dpi=dpi,
                   facecolor='white',
                   edgecolor='none')
        plt.close()
        
        print(f"✓ Generated: {output_path}")
        
    except Exception as e:
        print(f"✗ Error processing {audio_path}: {str(e)}")

def main():
    """Main function to generate all spectrograms."""
    
    # Define input and output directories
    input_dir = Path("input_wsj0c3")
    output_dir = Path("spectrograms")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' not found!")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Method directories to process
    methods = ['clean', 'noisy', 'sgmsep', 'vpidm', 'flowse', 'vpvid_sde', 'vpvid_sdec', 'vpvid_ode']
    
    print("Generating spectrograms...")
    print(f"Output size: 360x200 pixels")
    print("-" * 50)
    
    total_files = 0
    processed_files = 0
    
    # Process each method directory
    for method in methods:
        method_dir = input_dir / method
        if not method_dir.exists():
            print(f"Warning: Method directory '{method}' not found, skipping...")
            continue
        
        # Create output subdirectory for this method
        output_method_dir = output_dir / method
        output_method_dir.mkdir(exist_ok=True)
        
        # Process all wav files in the method directory
        wav_files = list(method_dir.glob("*.wav"))
        total_files += len(wav_files)
        
        print(f"\nProcessing {method} ({len(wav_files)} files):")
        
        for wav_file in wav_files:
            # Create output filename (change extension to .png)
            output_file = output_method_dir / f"{wav_file.stem}.png"
            
            # Generate spectrogram
            create_spectrogram(wav_file, output_file)
            processed_files += 1
    
    print("-" * 50)
    print(f"Processing complete!")
    print(f"Total files processed: {processed_files}/{total_files}")
    print(f"Spectrograms saved in: {output_dir}/")
    
    # Update HTML with correct paths
    print("\nTo use in HTML, update image src paths from:")
    print("  https://via.placeholder.com/360x200/f0f0f0/666?text=Spectrogram+Placeholder")
    print("To:")
    print("  spectrograms/{method}/{filename}.png")
    print("\nExample:")
    print("  spectrograms/clean/051o0211.png")
    print("  spectrograms/vpvid_sde/051o0211.png")

if __name__ == "__main__":
    main()