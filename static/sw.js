// Clean Air Walk service worker: makes the app installable and shows the last forecast offline.
const VERSION = 'caw-v1';
const SHELL = ['/', '/manifest.webmanifest', '/static/icons/icon-192.png', '/static/icons/icon-512.png',
               'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(VERSION).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => k !== VERSION).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;               // Gemma (/api/ask) and sky check need the network
  const url = new URL(req.url);
  // Forecast + page: network first, fall back to the last saved copy when offline.
  if (url.pathname === '/' || url.pathname.startsWith('/api/forecast')) {
    e.respondWith(fetch(req).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(VERSION).then(c => c.put(req, copy)); }
      return res;
    }).catch(() => caches.match(req).then(hit => hit || (url.pathname === '/'
      ? caches.match('/') : new Response(JSON.stringify({detail: 'offline and no saved forecast for this place'}),
        {status: 503, headers: {'Content-Type': 'application/json'}})))));
    return;
  }
  // Static assets and libraries: cache first.
  if (url.pathname.startsWith('/static/') || url.hostname === 'cdnjs.cloudflare.com' || url.hostname.includes('fonts.')) {
    e.respondWith(caches.match(req).then(hit => hit || fetch(req).then(res => {
      if (res.ok || res.type === 'opaque') { const copy = res.clone(); caches.open(VERSION).then(c => c.put(req, copy)); }
      return res;
    })));
  }
});
