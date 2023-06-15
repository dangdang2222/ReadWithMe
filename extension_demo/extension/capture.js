chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.requestImageCaptureBack) {
      console.log("capture got from back");
      captureImage();
    }
  });

function captureImage() {
  const canvas = document.createElement('canvas');
  canvas.width = captureVideo.videoWidth;
  canvas.height = captureVideo.videoHeight;
  canvas.getContext('2d').drawImage(captureVideo, 0, 0);
  const imageData = canvas.toDataURL();
  chrome.runtime.sendMessage({imageData: imageData});
};

const startCaptureButton = document.querySelector('#startCaptureButton');
const captureVideo = document.querySelector('#captureVideo');
let captureTimer = null;
let mediaStream = null;

startCaptureButton.addEventListener('click', async () => {
  if (!mediaStream) {
    mediaStream = await navigator.mediaDevices.getDisplayMedia({ video: { cursor: 'never' } });
    captureVideo.srcObject = mediaStream;
    captureVideo.play();
  }
});






  