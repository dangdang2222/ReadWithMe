chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.requestImageCapture) { 
      console.log("Back got from content");
      chrome.runtime.sendMessage({requestImageCaptureBack: true});
    } else if (request.imageData) {
      console.log('Back got from capture');
      chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        chrome.tabs.sendMessage(tabs[0].id, {imageData: request.imageData});
      });
    }
  });
  
  




  

  











  