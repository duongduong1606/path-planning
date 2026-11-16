// Simple cache-busting middleware for static file server
// Reads file modification time and appends ?t=<timestamp> to URL

const fs = require('fs');
const path = require('path');
const http = require('http');

const UI_DIR = path.join(__dirname);
const CACHE_DURATION = 3600000; // 1 hour for CSS
const NO_CACHE = ['app.js', 'style.css'];

const server = http.createServer((req, res) => {
    if (req.url === '/') {
        req.url = '/index.html';
    }
    
    if (!req.url.startsWith('/')) {
        return res.writeHead(404).end('Not found');
    }

    const filePath = path.join(UI_DIR, req.url);
    const ext = path.extname(filePath);

    const mimeTypes = {
        '.html': 'text/html',
        '.js': 'application/javascript',
        '.css': 'text/css',
        '.json': 'application/json',
    };

    if (!mimeTypes[ext]) {
        return res.writeHead(404).end('Not found');
    }

    fs.stat(filePath, (err, stats) => {
        if (err) {
            return res.writeHead(404).end('Not found');
        }

        const mtime = stats.mtimeMs;
        const cacheHeader = NO_CACHE.includes(req.url.split('/').pop()) 
            ? 'no-cache, no-store, must-revalidate' 
            : 'public, max-age=3600';

        const cacheBust = NO_CACHE.includes(req.url.split('/').pop())
            ? `?t=${Date.now()}`
            : '';

        const finalUrl = req.url === '/' ? '/index.html' : req.url + cacheBust;
        const finalPath = path.join(UI_DIR, finalUrl);

        fs.readFile(finalPath, (err, data) => {
            if (err) {
                return res.writeHead(500).end('Server error');
            }

            res.writeHead(200, {
                'Content-Type': mimeTypes[ext],
                'Cache-Control': cacheHeader,
                'ETag': '"' + mtime + '"'
            });

            res.end(data);
        });
    });
});

const PORT = process.env.PORT || 8765;
server.listen(PORT, '127.0.0.1', () => {
    console.log(`UI server running at http://127.0.0.1:${PORT}/`);
    console.log('app.js will be served with no-cache header');
});
