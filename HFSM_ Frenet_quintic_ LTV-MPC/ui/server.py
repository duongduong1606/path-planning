"""UI server with aggressive cache-busting for app.js"""
import http.server
import socketserver
import os
from urllib.parse import urlparse

PORT = int(os.environ.get('PORT', 8765))

# Get absolute path to repo root (where __main__.py is)
# __file__ is ui/server.py, so dirname is ui/, then dirname again is repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

NO_CACHE_FILES = frozenset(['app.js'])

class CacheBustingHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        filename = os.path.basename(parsed.path)
        
        # For no-cache files, add timestamp to force reload
        if filename in NO_CACHE_FILES:
            # Split on ? first, then join with ROOT_DIR
            path_parts = parsed.path.split('?')
            # Strip leading slash and drive letter for proper path joining
            clean_path = path_parts[0].lstrip('/').replace('C:', 'C:\\')
            base_path = os.path.abspath(os.path.join(ROOT_DIR, clean_path))
            
            # Verify file exists before proceeding
            if not os.path.exists(base_path):
                self.send_error(404, f"File not found: {base_path}")
                return
            
            timestamp = int(os.path.getmtime(base_path))
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/javascript')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('ETag', f'"{timestamp}"')
            self.end_headers()
            
            try:
                with open(base_path, 'rb') as f:
                    self.wfile.write(f.read())
            except Exception as e:
                self.send_error(404, f"File not found: {base_path}")
            return
        
        return super().do_GET()

class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True

if __name__ == '__main__':
    with ThreadingTCPServer(("", PORT), CacheBustingHandler) as httpd:
        print(f"UI server running at http://127.0.0.1:{PORT}/")
        print(f"Serving from: {ROOT_DIR}")
        print("app.js will be served with no-cache, no-store, must-revalidate")
        httpd.serve_forever()
