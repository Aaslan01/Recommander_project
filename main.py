#!/usr/bin/env python3
"""
🎬 Recommander Project - Main Entry Point

This is the main entry point for the Recommander project.
It provides a simple interface to run the complete pipeline.

Usage:
    python main.py                    # Run complete pipeline
    python main.py --help            # Show help
    python main.py --sentiment       # Generate sentiment data only
    python main.py --experiments     # Run experiments only
    python main.py --plots           # Generate plots only
"""

import sys
import os
import argparse
from pathlib import Path

def main():
    """Main entry point for the Recommander project."""
    
    parser = argparse.ArgumentParser(
        description="🎬 Recommander Project - Hybrid Movie Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Run complete pipeline
    python main.py --sentiment       # Generate sentiment data only
    python main.py --experiments     # Run experiments only
    python main.py --plots           # Generate plots only
        """
    )
    
    parser.add_argument(
        '--sentiment',
        action='store_true',
        help='Generate sentiment data only'
    )
    
    parser.add_argument(
        '--experiments',
        action='store_true',
        help='Run experiments only'
    )
    
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate plots only'
    )
    
    parser.add_argument(
        '--pipeline',
        action='store_true',
        help='Run complete pipeline (default)'
    )
    
    args = parser.parse_args()
    
    # If no specific option is chosen, default to pipeline
    if not any([args.sentiment, args.experiments, args.plots, args.pipeline]):
        args.pipeline = True
    
    print("🎬 Recommander Project")
    print("=" * 50)
    
    try:
        if args.sentiment:
            print("📊 Generating sentiment data...")
            os.system("python sentiment_analysis.py")
            
        elif args.experiments:
            print("🧪 Running experiments...")
            os.system("python run_experiments.py")
            
        elif args.plots:
            print("📈 Generating plots...")
            os.system("python plot_runner.py")
            
        elif args.pipeline:
            print("🚀 Running complete pipeline...")
            os.system("python final_pipeline.py")
            
        print("\n✅ Operation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Please check that all dependencies are installed and data files are present.")
        sys.exit(1)

if __name__ == "__main__":
    main() 