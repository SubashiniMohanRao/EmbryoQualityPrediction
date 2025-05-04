import os
import sys
import argparse
import subprocess
import webbrowser
from pathlib import Path

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

def main():
    """Main function to evaluate models and launch the visualization dashboard."""
    parser = argparse.ArgumentParser(description='Evaluate models and launch visualization dashboard')
    parser.add_argument('--model', type=str, help='Path to specific model file to evaluate')
    parser.add_argument('--evaluate_only', action='store_true', help='Only run evaluation without launching dashboard')
    parser.add_argument('--dashboard_only', action='store_true', help='Only launch dashboard without evaluation')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the dashboard on')
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "results"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "plots"), exist_ok=True)
    
    # Run evaluation if requested
    if not args.dashboard_only:
        print("=" * 80)
        print("STEP 1: Running model evaluation")
        print("=" * 80)
        
        eval_cmd = [sys.executable, os.path.join(PROJECT_ROOT, "src", "evaluate_model.py")]
        if args.model:
            eval_cmd.extend(["--model", args.model])
        
        try:
            subprocess.run(eval_cmd, check=True)
            print("\n[SUCCESS] Model evaluation completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Error during model evaluation: {e}")
            if not args.dashboard_only:
                sys.exit(1)
    
    # Launch dashboard if requested
    if not args.evaluate_only:
        print("\n" + "=" * 80)
        print("STEP 2: Launching visualization dashboard")
        print("=" * 80)
        
        dashboard_url = f"http://localhost:{args.port}"
        print(f"\n[INFO] Dashboard will be available at: {dashboard_url}")
        print("[INFO] Opening web browser automatically...")
        
        # Start the Flask app in a new process
        flask_cmd = [
            sys.executable, 
            os.path.join(PROJECT_ROOT, "app", "app.py")
        ]
        
        # Set environment variables for Flask
        env = os.environ.copy()
        env["FLASK_APP"] = os.path.join(PROJECT_ROOT, "app", "app.py")
        env["FLASK_ENV"] = "development"
        
        # Open browser after a short delay
        import threading
        import time
        
        def open_browser():
            time.sleep(2)  # Wait for Flask to start
            webbrowser.open(dashboard_url)
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run Flask app
        try:
            subprocess.run(flask_cmd, env=env, check=True)
        except KeyboardInterrupt:
            print("\n[INFO] Dashboard server stopped")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Error launching dashboard: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
