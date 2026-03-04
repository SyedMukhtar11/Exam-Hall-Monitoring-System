const img = document.getElementById("videoStream");
const alertText = document.getElementById("alertText");
const alarm = document.getElementById("alarmSound");

let alarmPlaying = false;

// 🎥 Webcam
function startWebcam() {
    img.src = "http://127.0.0.1:8000/webcam";
}

// 📁 Upload Video
async function uploadVideo() {
    const fileInput = document.getElementById("videoFile");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select a video");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    await fetch("http://127.0.0.1:8000/upload/", {
        method: "POST",
        body: formData
    });

    // ✅ SHOW PROCESSED VIDEO (NOT RAW FILE)
    img.src = "http://127.0.0.1:8000/video";
}

// 🚨 REAL-TIME ALERT SYNC
setInterval(async () => {
    const res = await fetch("http://127.0.0.1:8000/alert");
    const data = await res.json();

    if (data.alert) {
        alertText.innerText = "🚨 CHEATING DETECTED!";
        
        if (!alarmPlaying) {
            alarm.play();
            alarmPlaying = true;
        }

    } else {
        alertText.innerText = "No cheating detected";
        alarm.pause();
        alarm.currentTime = 0;
        alarmPlaying = false;
    }

}, 1000);