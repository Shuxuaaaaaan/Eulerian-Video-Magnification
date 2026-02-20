import time
import os
import subprocess
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

# Define paths
WATCH_DIR = "/app/data/resources"
OUTPUT_DIR = "/app/data/results"
EVM_SCRIPT = "/app/src/evm.py"

class VideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Ignore directories and non-mp4 files
        if event.is_directory or not event.src_path.lower().endswith('.mp4'):
            return
            
        print(f"[Watcher] New video detected: {event.src_path}", flush=True)
        self.process_video(event.src_path)
        
    def process_video(self, input_path):
        filename = os.path.basename(input_path)
        output_path = os.path.join(OUTPUT_DIR, f"magnified_{filename}")
        
        # Check if already processed to avoid infinite loops if results are in the same or monitored folder
        if os.path.exists(output_path):
            print(f"[Watcher] Video {filename} was already processed. Skipping.", flush=True)
            return

        print(f"[Watcher] Starting processing for {filename}...", flush=True)
        start_time = time.time()
        
        try:
            # Call the main script externally to isolate memory and avoid threading issues
            # Using CPU mode with 4 threads by default (adjustable or via env vars)
            cmd = [
                "/app/.venv/bin/python", EVM_SCRIPT,
                "-v", input_path,
                "-s", output_path,
                "--accel", "cpu",
                "-t", "4"
            ]
            
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            print(f"[Watcher] Successfully processed {filename} in {time.time() - start_time:.2f}s", flush=True)
            
        except subprocess.CalledProcessError as e:
            print(f"[Watcher] Error processing {filename}:", flush=True)
            print(e.stderr, flush=True)
        except Exception as e:
             print(f"[Watcher] Unexpected error processing {filename}: {e}", flush=True)

if __name__ == "__main__":
    try:
        # Ensure directories exist
        os.makedirs(WATCH_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"[Watcher] Starting Eulerian Video Magnification Daemon", flush=True)
        print(f"[Watcher] Monitoring directory: {WATCH_DIR}", flush=True)
        
        event_handler = VideoHandler()
        
        # Initial scan to catch any files added while the container was off
        print(f"[Watcher] Performing initial scan...", flush=True)
        for f in os.listdir(WATCH_DIR):
            if f.lower().endswith('.mp4'):
                event_handler.process_video(os.path.join(WATCH_DIR, f))

        # Start listening for new files
        observer = PollingObserver()
        observer.schedule(event_handler, WATCH_DIR, recursive=False)
        observer.start()
        
        print(f"[Watcher] Initial scan complete. Listening for new files...", flush=True)
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        observer.stop()
        print("[Watcher] Shutting down cleanly...", flush=True)
    except Exception as e:
        import traceback
        print(f"[Watcher] FATAL CRASH: {e}", flush=True)
        traceback.print_exc()
        # Force a sleep so Docker doesn't instantly restart it in a tight loop and flood logs
        time.sleep(5)
        raise e
    finally:
        if 'observer' in locals():
            observer.join()
