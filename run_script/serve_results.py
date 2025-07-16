#!/usr/bin/env python3
"""
Simple HTTP server to serve trajectory analysis results.
Useful for viewing HTML files on remote machines.
"""

import argparse
import http.server
import socketserver
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Start a simple HTTP server to view trajectory analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Serve from a trajectory results folder
  python run_script/serve_results.py --folder-path path/to/trajectory/folder
  
  # Serve on a specific port
  python run_script/serve_results.py --folder-path path/to/folder --port 8080
        """
    )
    
    parser.add_argument(
        '--folder-path',
        required=True,
        help='Path to the folder containing analysis_viewer.html and trajectory_analysis_results.json'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to serve on (default: 8000)'
    )
    
    args = parser.parse_args()
    
    # Validate folder path
    if not os.path.exists(args.folder_path):
        print(f"Error: Folder path does not exist: {args.folder_path}")
        sys.exit(1)
    
    # Check for required files
    html_file = os.path.join(args.folder_path, "analysis_viewer.html")
    json_file = os.path.join(args.folder_path, "trajectory_analysis_results.json")
    
    if not os.path.exists(html_file):
        print(f"Warning: analysis_viewer.html not found in {args.folder_path}")
        print("Make sure you've run the analysis script first.")
    
    if not os.path.exists(json_file):
        print(f"Warning: trajectory_analysis_results.json not found in {args.folder_path}")
        print("Make sure you've run the analysis script first.")
    
    # Change to the target directory
    os.chdir(args.folder_path)
    
    # Create and start the server
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", args.port), Handler) as httpd:
            print(f"🌐 Serving trajectory analysis results at:")
            print(f"   http://localhost:{args.port}")
            print(f"   http://localhost:{args.port}/analysis_viewer.html")
            print()
            print(f"📂 Serving files from: {args.folder_path}")
            print("Press Ctrl+C to stop the server")
            print()
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {args.port} is already in use. Try a different port:")
            print(f"   python run_script/serve_results.py --folder-path {args.folder_path} --port {args.port + 1}")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()