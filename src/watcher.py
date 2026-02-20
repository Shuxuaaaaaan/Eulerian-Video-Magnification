import time
import os
import subprocess
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

# Define paths
WATCH_DIR = "/app/data/resources"
OUTPUT_DIR = "/app/data/results"
EVM_SCRIPT = "/app/src/evm.py"

# Configurable via environment variables (set in docker-compose.yml)
EVM_THREADS = os.environ.get("EVM_THREADS", "1")
EVM_ACCEL = os.environ.get("EVM_ACCEL", "cpu")
EVM_MODE = os.environ.get("EVM_MODE")             # gaussian / laplacian
EVM_LEVEL = os.environ.get("EVM_LEVEL")            # Pyramid levels
EVM_ALPHA = os.environ.get("EVM_ALPHA")            # Amplification factor
EVM_LAMBDA_CUTOFF = os.environ.get("EVM_LAMBDA_CUTOFF")  # Lambda cutoff
EVM_LOW_OMEGA = os.environ.get("EVM_LOW_OMEGA")    # Min frequency
EVM_HIGH_OMEGA = os.environ.get("EVM_HIGH_OMEGA")  # Max frequency
EVM_ATTENUATION = os.environ.get("EVM_ATTENUATION") # I/Q channel attenuation

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
            cmd = [
                "/app/.venv/bin/python", EVM_SCRIPT,
                "-v", input_path,
                "-s", output_path,
                "--accel", EVM_ACCEL,
                "-t", EVM_THREADS
            ]
            # Append optional algorithm parameters if configured
            if EVM_MODE:            cmd += ["-m", EVM_MODE]
            if EVM_LEVEL:           cmd += ["-l", EVM_LEVEL]
            if EVM_ALPHA:           cmd += ["-a", EVM_ALPHA]
            if EVM_LAMBDA_CUTOFF:   cmd += ["-lc", EVM_LAMBDA_CUTOFF]
            if EVM_LOW_OMEGA:       cmd += ["-lo", EVM_LOW_OMEGA]
            if EVM_HIGH_OMEGA:      cmd += ["-ho", EVM_HIGH_OMEGA]
            if EVM_ATTENUATION:     cmd += ["-at", EVM_ATTENUATION]
            
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            print(f"[Watcher] Successfully processed {filename} in {time.time() - start_time:.2f}s", flush=True)
            
        except subprocess.CalledProcessError as e:
            print(f"[Watcher] Error processing {filename} (exit code {e.returncode}):", flush=True)
            if e.stdout:
                print(f"[Watcher] STDOUT:\n{e.stdout}", flush=True)
            if e.stderr:
                # Filter out tqdm progress bars to show only the real error
                error_lines = [line for line in e.stderr.splitlines() 
                               if not line.strip().startswith(('Gaussian', 'Laplacian', '%|', ''))]
                real_error = '\n'.join(error_lines) if error_lines else e.stderr[-500:]
                print(f"[Watcher] STDERR:\n{real_error}", flush=True)
        except Exception as e:
             print(f"[Watcher] Unexpected error processing {filename}: {e}", flush=True)

if __name__ == "__main__":
    try:
        # Ensure directories exist
        os.makedirs(WATCH_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"[Watcher] Starting Eulerian Video Magnification Daemon", flush=True)
        print(f"[Watcher] Monitoring directory: {WATCH_DIR}", flush=True)
        print(f"[Watcher] Config: accel={EVM_ACCEL}, threads={EVM_THREADS}", flush=True)
        
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
