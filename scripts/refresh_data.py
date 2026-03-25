#!/usr/bin/env python3
"""
Weekly database refresh script
Can be run manually or via cron job
"""

import pandas as pd
import requests
import os
from datetime import datetime
import sys

def refresh_database():
    """Download and update database CSV"""
    
    print(f"[{datetime.now()}] Starting database refresh...")
    
    try:
        # Method 1: Download from URL (configure in environment)
        csv_url = os.getenv('CSV_SOURCE_URL')
        
        if csv_url:
            print(f"Downloading from: {csv_url}")
            response = requests.get(csv_url, timeout=30)
            response.raise_for_status()
            
            # Save with timestamp
            timestamp = datetime.now().strftime('%Y%m%d')
            output_path = f'data/az_database_list_{timestamp}.csv'
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Downloaded to: {output_path}")
        
        else:
            # Method 2: Copy existing file with new timestamp
            timestamp = datetime.now().strftime('%Y%m%d')
            source = 'data/az_database_list_export_2026_02_24_cleaned.csv'
            dest = f'data/az_database_list_{timestamp}.csv'
            
            import shutil
            shutil.copy(source, dest)
            print(f"✓ Copied {source} to {dest}")
        
        # Validate the CSV
        df = pd.read_csv(output_path if csv_url else dest)
        print(f"✓ Validation passed: {len(df)} databases loaded")
        
        if len(df) < 100:
            print("⚠️  Warning: Database count seems unusually low")
        
        # Clean old files (keep last 4 weeks)
        cleanup_old_files()
        
        print(f"[{datetime.now()}] ✓ Database refresh completed successfully")
        return True
    
    except Exception as e:
        print(f"❌ Error during refresh: {e}")
        sys.exit(1)

def cleanup_old_files():
    """Remove CSV files older than 4 weeks"""
    import glob
    from datetime import timedelta
    
    files = glob.glob('data/az_database_list_*.csv')
    cutoff = datetime.now() - timedelta(weeks=4)
    
    for file in files:
        file_time = datetime.fromtimestamp(os.path.getctime(file))
        if file_time < cutoff:
            os.remove(file)
            print(f"Cleaned up old file: {file}")

if __name__ == "__main__":
    refresh_database()
