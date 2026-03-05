// ✅ Correct element references
const videoStream = document.getElementById("videoStream");
const alertText = document.getElementById("alertText");
const alarm = document.getElementById("alarmSound");
let alarmPlaying = false;

// 🎥 Webcam — assign src directly to <img>
function startWebcam() {
    videoStream.src = "http://127.0.0.1:8000/webcam";
}

// 📁 Upload Video
async function uploadVideo() {
    const fileInput = document.getElementById("videoFile");
    const file = fileInput.files[0];
    if (!file) {
        alert("Please select a video");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:8000/upload/", {
        method: "POST",
        body: formData
    });

    if (!res.ok) {
        alert("Upload failed");
        return;
    }

    // ✅ Backend /video route takes no filename — it uses last uploaded
    videoStream.src = "http://127.0.0.1:8000/video";
}

// 🚨 Real-time alert polling
setInterval(async () => {
    try {
        const res = await fetch("http://127.0.0.1:8000/alert");
        const data = await res.json();

        if (data.alert) {
            alertText.innerText = "🚨 CHEATING DETECTED! Count: " + data.count;
            if (!alarmPlaying) {
                alarm.play();
                alarmPlaying = true;
            }
        } else {
            alertText.innerText = "✅ No cheating detected";
            alarm.pause();
            alarm.currentTime = 0;
            alarmPlaying = false;
        }
    } catch (err) {
        console.error("Alert fetch error:", err);
    }
}, 1000);