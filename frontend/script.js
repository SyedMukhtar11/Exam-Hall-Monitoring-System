const img = document.getElementById("videoStream");
const alertText = document.getElementById("alertText");
const alarm = document.getElementById("alarmSound");

// 🎥 Webcam
function startWebcam() {
    img.src = "http://127.0.0.1:8000/webcam";
}

// 📁 Upload Video
async function uploadVideo() {
    const fileInput = document.getElementById("videoFile");
    const file = fileInput.files[0];

    let formData = new FormData();
    formData.append("file", file);

    await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData
    });

    img.src = `http://127.0.0.1:8000/video/${file.name}`;
}

// 🚨 ALERT CHECK LOOP
setInterval(async () => {
    const res = await fetch("http://127.0.0.1:8000/alert");
    const data = await res.json();

    if (data.alert) {
        alertText.innerText = "🚨 CHEATING DETECTED!";
        alarm.play();
    } else {
        alertText.innerText = "No cheating detected";
    }
}, 2000);