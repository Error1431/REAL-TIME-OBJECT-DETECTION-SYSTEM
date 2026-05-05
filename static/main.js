// Elements
const fpsEl = document.getElementById('fps-val');
const trackCountEl = document.getElementById('track-count-val');
const entriesEl = document.getElementById('entries-val');
const exitsEl = document.getElementById('exits-val');
const skipValEl = document.getElementById('skip-val');
const confDisplay = document.getElementById('conf-display');

// Inputs
const confSlider = document.getElementById('confidence');
const skipSelect = document.getElementById('skip-frames');
const imgszSelect = document.getElementById('imgsz');
const sourceInput = document.getElementById('video-source');
const filterInput = document.getElementById('class-filters');
const btnSource = document.getElementById('btn-update-source');
const btnFilter = document.getElementById('btn-update-filter');

let initialLoadDone = false; // Add flag so we don't overwrite inputs immediately if user is typing

// Polling interval for live stats (e.g. 500ms)
setInterval(fetchStats, 500);

async function fetchStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        
        fpsEl.innerText = data.fps.toFixed(1);
        trackCountEl.innerText = data.track_count;
        entriesEl.innerText = data.entries;
        exitsEl.innerText = data.exits;
        skipValEl.innerText = data.skip_frames;

        
        if (!initialLoadDone) {
            sourceInput.value = data.source;
            filterInput.value = data.target_classes.join(", ");
            initialLoadDone = true;
        }
        
    } catch (e) {
        console.error("Failed to fetch stats", e);
    }
}

// Function to push settings back to FastAPI
async function updateSettings() {
    let classesRaw = filterInput.value.split(',').map(s => s.trim()).filter(s => s !== "");
    
    const payload = {
        confidence: parseFloat(confSlider.value),
        skip_frames: parseInt(skipSelect.value),
        imgsz: parseInt(imgszSelect.value),
        source: sourceInput.value,
        target_classes: classesRaw
    };
    
    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
    } catch (e) {
        console.error("Failed to post settings update", e);
    }
}

// Event Listeners for UI interaction
confSlider.addEventListener('input', (e) => {
    confDisplay.innerText = parseFloat(e.target.value).toFixed(2);
});

confSlider.addEventListener('change', updateSettings);
skipSelect.addEventListener('change', updateSettings);
imgszSelect.addEventListener('change', updateSettings);
btnSource.addEventListener('click', updateSettings);
btnFilter.addEventListener('click', updateSettings);

// Fetch initial stats immediately to sync UI knobs
fetchStats();

// --- Chart setup ---
const ctx = document.getElementById('analyticsChart').getContext('2d');
const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Entries',
                data: [],
                borderColor: '#10b981', // emerald-500
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                fill: true,
                tension: 0.3
            },
            {
                label: 'Exits',
                data: [],
                borderColor: '#f43f5e', // rose-500
                backgroundColor: 'rgba(244, 63, 94, 0.1)',
                fill: true,
                tension: 0.3
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { labels: { color: '#c9d1d9' } }
        },
        scales: {
            x: { ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
            y: { ticks: { color: '#8b949e', stepSize: 1 }, grid: { color: '#30363d' } }
        }
    }
});

// Update chart every 10 seconds
setInterval(async () => {
    try {
        const res = await fetch('/api/history?minutes=60');
        const data = await res.json();
        
        chart.data.labels = data.labels;
        chart.data.datasets[0].data = data.entries;
        chart.data.datasets[1].data = data.exits;
        chart.update();
    } catch (e) {
        console.error("Failed to fetch history API", e);
    }
}, 10000);

// Kick off first chart load
setTimeout(() => {
    fetch('/api/history?minutes=60')
        .then(r => r.json())
        .then(data => {
            chart.data.labels = data.labels;
            chart.data.datasets[0].data = data.entries;
            chart.data.datasets[1].data = data.exits;
            chart.update();
        });
}, 500);
