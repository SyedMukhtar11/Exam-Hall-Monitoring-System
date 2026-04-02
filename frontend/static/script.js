// ── Element References ────────────────────────────────────────
const videoStream = document.getElementById("videoStream");
const alertBanner = document.getElementById("alertBanner");
const alertBox    = document.getElementById("alertBox");
const alertText   = document.getElementById("alertText");
const alertCount  = document.getElementById("alertCount");
const alarm       = document.getElementById("alarmSound");

let alarmPlaying  = false;
let pollInterval  = null;


// ── Alert Helpers ─────────────────────────────────────────────
function showAlert(count) {
    // Show flashing red banner
    alertBanner.classList.remove("hidden");

    // Style the status box as danger
    alertBox.classList.remove("safe");
    alertBox.classList.add("danger");

    // Update text
    alertText.style.color = "#ff4444";
    alertText.innerText   = "🚨 CHEATING DETECTED!";
    alertCount.innerText  = `Suspicious frames flagged: ${count}`;

    // Play alarm if not already playing
    if (!alarmPlaying) {
        alarm.currentTime = 0;
        alarm.play().catch(err => console.warn("⚠ Alarm play blocked by browser:", err));
        alarmPlaying = true;
    }
}

function clearAlert() {
    // Hide banner
    alertBanner.classList.add("hidden");

    // Style the status box as safe
    alertBox.classList.remove("danger");
    alertBox.classList.add("safe");

    // Update text
    alertText.style.color = "#00ff88";
    alertText.innerText   = "✅ No cheating detected";
    alertCount.innerText  = "";

    // Stop alarm
    if (alarmPlaying) {
        alarm.pause();
        alarm.currentTime = 0;
        alarmPlaying = false;
    }
}


// ── Start Polling /alert ──────────────────────────────────────
function startPolling() {
    // Clear any existing interval first
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const res  = await fetch("http://127.0.0.1:8000/alert");
            const data = await res.json();

            if (data.alert) {
                showAlert(data.count);
            } else {
                clearAlert();
            }
        } catch (err) {
            console.error("Alert poll error:", err);
        }
    }, 1000);
}


// ── Webcam Stream ─────────────────────────────────────────────
function startWebcam() {
    // Cache-bust so repeated clicks always reconnect
    videoStream.src = `http://127.0.0.1:8000/webcam?t=${Date.now()}`;
    videoStream.alt = "Loading webcam...";
    startPolling();
}


// ── Upload + Video Stream ─────────────────────────────────────
async function uploadVideo() {
    const fileInput = document.getElementById("videoFile");
    const file      = fileInput.files[0];

    if (!file) {
        alert("⚠ Please select a video file first.");
        return;
    }

    // Show uploading feedback
    alertText.style.color = "#00c2ff";
    alertText.innerText   = "⏳ Uploading video...";
    alertCount.innerText  = "";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("http://127.0.0.1:8000/upload/", {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            alert("❌ Upload failed. Please try again.");
            clearAlert();
            return;
        }

        // Start streaming the uploaded video with detection
        videoStream.src = `http://127.0.0.1:8000/video?t=${Date.now()}`;
        videoStream.alt = "Loading video stream...";
        startPolling();

    } catch (err) {
        console.error("Upload error:", err);
        alert("❌ Upload error: " + err.message);
    }
}


// ── Stop Stream ───────────────────────────────────────────────
function stopStream() {
    // Clear the video
    videoStream.src = "";
    videoStream.alt = "Stream stopped.";

    // Stop polling
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }

    // Reset alert UI
    clearAlert();
    alertText.innerText  = "⏹ Stream stopped.";
    alertText.style.color = "#aaaaaa";
}