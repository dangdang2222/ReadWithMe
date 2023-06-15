let clicked_x , clicked_y;
let coordinates;
let scrolling;
let interval;
let ten_coordinates;
let browsing_flag = 1;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.imageData) {
    console.log("content got from back");
    const image = new Image();
    image.src = request.imageData;

    sendToOCRServer(request.imageData, coordinates);
  }
});

window.addEventListener('scroll', (e) => {
  if (!scrolling) {
    console.log('start scrolling!');
  }
  
  // If there is a pending interval, clear it
  if (interval) {
    clearInterval(interval);
    interval = undefined;
  }

  // After a short delay, check if scrolling has stopped
  clearTimeout(scrolling);
  scrolling = setTimeout(() => {
    console.log('stop scrolling!');
    
    // Every 5 seconds after stopping scrolling, output a message
    interval = setInterval(() => {
      console.log('stopped for 5secs');
      

      if(browsing_flag == 1){
        chrome.runtime.sendMessage({requestImageCapture: true});
      }
        
    }, 5000);

    scrolling = undefined;
  }, 250);
});





function sendToOCRServer(imageData, _coordinates) {
  const resultsText = document.getElementById('resultsText');
  resultsText.textContent = "";

  // Convert image data to base64 format
  const imageDataBase64 = imageData.replace(/^data:image\/(png|jpeg|jpg);base64,/, '');

  const coordinatesString = JSON.stringify(_coordinates);
  
  // Send the image data and coordinates to the OCR server
  fetch('http://127.0.0.1:8000/extract', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: `image=${encodeURIComponent(imageDataBase64)}&coordinates=${encodeURIComponent(coordinatesString)}`,
  })
    .then(response => response.json())
    .then(data => {
      // Handle the response from the OCR server
      console.log(data);
      resultsText.insertAdjacentHTML('beforeend', data.answer);
      toggleSmallWindow(true);
    })
    .catch(error => {
      console.error('Error:', error);
    });
}

async function sendImageToGazeServer(imageDataURL) {
  const response = await fetch('http://127.0.0.1:5000/process_image', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: `image=${encodeURIComponent(imageDataURL)}`,
  });

  // Parse the response as JSON
  const data = await response.json();
  console.log(data);
  return data.coordinates;
}


const sse = new EventSource('http://127.0.0.1:8000/stream');

sse.addEventListener('gpt_response', function(event) {
  const resultsText = document.getElementById('resultsText');
  const data = JSON.parse(event.data);
  console.log(data);
  
  let message = data.message;
  
  // Check if the message is an empty string
  if (message === "") {
    message = ' '; // Replace it with a space character
  }
  
  resultsText.insertAdjacentHTML('beforeend', message);

  toggleSmallWindow(true);
});



function sendToGPTServer(_sentence) {

  console.log('GPTGPTSERVER');

  const resultsText = document.getElementById('resultsText');
  resultsText.textContent = "";


  const sentenceString = JSON.stringify(_sentence);


  
  // Send the image data and coordinates to the OCR server
  fetch('http://127.0.0.1:8000/answer', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: `sentence=${encodeURIComponent(sentenceString)}`,
  })
    .then(response => response.json())
    .then(data => {
      // Handle the response from the GPT server
      console.log(data.answer);
    })
    .catch(error => {
      console.error('Error:', error);
    });
}


async function captureWebcamImage() {
  const video = document.createElement('video');
  
  // Get webcam stream
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await video.play();

  // Capture current frame
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  // Stop video stream
  video.pause();
  stream.getTracks()[0].stop();

  // Return data URL of the captured image
  return canvas.toDataURL();
}


// Set an interval to send the webcam image every 5 seconds
setInterval(async function() {
  // Capture image from webcam
  const webcamImageDataURL = await captureWebcamImage();

  // Send the captured image data to the Python server
  coordinates = await sendImageToGazeServer(webcamImageDataURL, 0, 0); // x, y values are set to 0 as they are not relevant in this context

  // Update the small window with the received image data
}, 333); // 1000 milliseconds = 1 second




function createSmallWindow() {
  const smallWindow = document.createElement('div');
  smallWindow.id = 'extensionSmallWindow';
  smallWindow.style.position = 'fixed';
  smallWindow.style.zIndex = '9999';
  smallWindow.style.right = '20px';
  smallWindow.style.top = '20px';
  smallWindow.style.width = '300px';
  smallWindow.style.height = '200px';
  smallWindow.style.backgroundColor = 'white';
  smallWindow.style.padding = '33px';
  smallWindow.style.borderRadius = '4px';
  smallWindow.style.overflowY = 'auto';
  smallWindow.style.boxShadow = '0px 0px 10px rgba(0, 0, 0, 0.3)';
  smallWindow.style.display = 'block';
  smallWindow.style.fontSize = '3px';

  const closeSmallWindowButton = document.createElement('button');
  closeSmallWindowButton.id = 'closeSmallWindowButton';
  closeSmallWindowButton.textContent = '닫고 탐색';
  closeSmallWindowButton.style.position = 'absolute';
  closeSmallWindowButton.style.top = '5px';
  closeSmallWindowButton.style.right = '5px';
  closeSmallWindowButton.style.display = 'block';

  const sendButton = document.createElement('button');
  sendButton.id = 'sendButton';
  sendButton.textContent = '질문';
  sendButton.style.position = 'absolute';
  sendButton.style.top = '5px';
  sendButton.style.right = '70px';
  sendButton.style.display = 'block';

  const resultsText = document.createElement('p');
  resultsText.id = 'resultsText';
  
  
  smallWindow.appendChild(sendButton);
  smallWindow.appendChild(closeSmallWindowButton);
  smallWindow.appendChild(resultsText);
  document.body.appendChild(smallWindow);

  return smallWindow;
}

const smallWindow = createSmallWindow();


function toggleSmallWindow(display) {
  smallWindow.style.display = display ? 'block' : 'none';
}

document.addEventListener('click', (event) => {
  if (event.target.id === 'closeSmallWindowButton') {
    toggleSmallWindow(false);
    browsing_flag = 1;
  }
});


document.addEventListener('click', (event) => {
  if (event.target.id === 'sendButton') {
    console.log('!!!!!!!!!!!!');
    browsing_flag = 0;
    sendToGPTServer(smallWindow.textContent);
  }
});





  
  
  
  
  
  
  