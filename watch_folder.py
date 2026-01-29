"""
Batch Input Folder Watcher

Monitors the batch_input folder and ensures only the most recent image is kept.
When a new image is added, all previous images are deleted.

Usage:
    python watch_folder.py              # Run continuously
    python watch_folder.py --once       # Clean once and exit
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_INPUT_DIR = os.path.join(SCRIPT_DIR, "batch_input")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CHECK_INTERVAL = 1  # seconds


def log(message: str):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def get_image_files() -> list:
    """Get all image files in batch_input folder, sorted by creation time (newest first).
    
    Uses creation time (getctime) on Windows to track when file was added to folder,
    not when it was originally created/modified elsewhere.
    """
    if not os.path.exists(BATCH_INPUT_DIR):
        return []
    
    images = []
    for filename in os.listdir(BATCH_INPUT_DIR):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            filepath = os.path.join(BATCH_INPUT_DIR, filename)
            # Use getctime (creation time on Windows = when file was added to this folder)
            ctime = os.path.getctime(filepath)
            images.append((filepath, filename, ctime))
    
    # Sort by creation time, newest first
    images.sort(key=lambda x: x[2], reverse=True)
    return images


def cleanup_old_images() -> bool:
    """Delete all images except the newest one. Returns True if any were deleted."""
    images = get_image_files()
    
    if len(images) <= 1:
        return False  # Nothing to clean up
    
    # Keep the first (newest), delete the rest
    newest = images[0]
    to_delete = images[1:]
    
    newest_time = datetime.fromtimestamp(newest[2]).strftime("%Y-%m-%d %H:%M:%S")
    log(f"✅ Keeping newest: {newest[1]} (added: {newest_time})")
    
    for filepath, filename, ctime in to_delete:
        file_time = datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
        try:
            os.remove(filepath)
            log(f"🗑️  Deleted: {filename} (added: {file_time})")
        except Exception as e:
            log(f"⚠️  Failed to delete {filename}: {e}")
    
    return True


def watch_folder():
    """Continuously watch the folder and clean up old images."""
    log("=" * 50)
    log("Batch Input Folder Watcher")
    log("=" * 50)
    log(f"Watching: {BATCH_INPUT_DIR}")
    log(f"Check interval: {CHECK_INTERVAL}s")
    log("Press Ctrl+C to stop")
    log("")
    
    # Initial cleanup
    cleanup_old_images()
    
    last_count = len(get_image_files())
    
    try:
        while True:
            time.sleep(CHECK_INTERVAL)
            
            current_images = get_image_files()
            current_count = len(current_images)
            
            # If new images were added (count increased)
            if current_count > last_count:
                log(f"📸 New image detected!")
                cleanup_old_images()
            
            last_count = len(get_image_files())  # Re-count after cleanup
            
    except KeyboardInterrupt:
        log("\n👋 Watcher stopped.")


def main():
    parser = argparse.ArgumentParser(description="Watch batch_input folder and keep only the newest image")
    parser.add_argument("--once", action="store_true", help="Clean up once and exit (don't watch)")
    args = parser.parse_args()
    
    if not os.path.exists(BATCH_INPUT_DIR):
        log(f"Creating batch_input folder: {BATCH_INPUT_DIR}")
        os.makedirs(BATCH_INPUT_DIR)
    
    if args.once:
        log("Running one-time cleanup...")
        if cleanup_old_images():
            log("Cleanup complete.")
        else:
            log("Nothing to clean up (0 or 1 images found).")
    else:
        watch_folder()


if __name__ == "__main__":
    main()
