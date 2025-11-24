// Open phone camera
function openCamera(){
  const cameraInput = document.getElementById("cameraInput");
  if(cameraInput) cameraInput.click();
}


// Full-screen preview
function openFullscreen(img) {
    const win = window.open("");
    win.document.write("<img src='" + img.src + "' style='width:100%'>");
}

// Download selected images
function downloadSelected() {
    const checked = [...document.querySelectorAll("input[name='selected']:checked")]
                    .map(e => e.value);

    if (checked.length === 0) {
        alert("Select at least 1 image!");
        return;
    }

    fetch("/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images: checked })
    })
    .then(res => res.blob())
    .then(blob => {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "similar_images.zip";
        a.click();
    });
}

// Feedback
function sendFeedback(image, relevant) {
    fetch("/feedback", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ image: image, feedback: relevant })
    }).then(() => alert("Feedback saved!"));
}
