const displayContainer = document.getElementById("display-container");
const streamImg = document.getElementById("stream"); // The <img> for live feed
const videoPlayer = document.getElementById("video-player"); // The <video> for uploads
const alarm = document.getElementById("alarm");

// 1. LIVE WEBCAM FIX
function webcam() {
    // Hide video player, show image for MJPEG stream
    videoPlayer.style.display = "none";
    streamImg.style.display = "block";
    
    // Add a cache-buster timestamp to force a fresh connection
    streamImg.src = "http://127.0.0.1" + new Date().getTime();
}

// 2. UPLOAD & GENERATED VIDEO FIX
async function upload() {
    let fileInput = document.getElementById("file");
    if (fileInput.files.length === 0) return alert("Select a file!");

    let fd = new FormData();
    // Use [0] to get the file object
    fd.append("file", fileInput.files[0]); 

    await fetch("http://127.0.0.1", {
        method: "POST",
        body: fd
    });

    // Change stream source
    document.getElementById("stream").src = "http://127.0.0.1" + new Date().getTime();
}




// Alert Polling (remains similar but wrapped in try/catch)
setInterval(async () => {
    try {
        let r = await fetch("/alert");
        let d = await r.json();

        if (d.alert) {
            document.getElementById("alert").innerText = d.msg;
            alarm.play();
        } else {
            document.getElementById("alert").innerText = "No cheating";
            alarm.pause();
            alarm.currentTime = 0;
        }
    } catch (e) { /* Backend might be busy */ }
}, 1000);
