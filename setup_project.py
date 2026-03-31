#!/usr/bin/env python
"""
Project setup utility - Creates required directories and generates sample data.

Usage:
    python setup_project.py
"""

import os
import pandas as pd
import sys

def create_directories():
    """Create required directory structure."""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'reports',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_sample_data():
    """Create sample ride-sharing data for testing."""
    sample_data = pd.DataFrame({
        'pickup_location': ['Downtown', 'Airport', 'Downtown', 'Suburb', 'Downtown', 'Airport'] * 20,
        'dropoff_location': ['Airport', 'Downtown', 'Suburb', 'Downtown', 'Airport', 'Suburb'] * 20,
        'hour_of_day': [8, 6, 15, 18, 22, 10] * 20,
        'day_of_week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'] * 20,
        'trip_distance': [15.2, 22.5, 4.3, 8.1, 18.5, 20.0] * 20,
        'estimated_time': [25, 35, 12, 20, 32, 30] * 20,
        'ride_completed': [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 10
    })
    
    # Save to CSV
    filepath = 'data/raw/ride_data.csv'
    sample_data.to_csv(filepath, index=False)
    print(f"✓ Created sample data: {filepath} ({len(sample_data)} rows)")


def main():
    """Run all setup steps."""
    print("=" * 60)
    print("ML Pipeline Project Setup")
    print("=" * 60)
    
    try:
        print("\n1. Creating directory structure...")
        create_directories()
        
        print("\n2. Generating sample data...")
        create_sample_data()
        
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. pip install -r requirements.txt")
        print("2. python -m src.main")
        print("3. pytest src/test_pipeline.py -v")
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during setup: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
