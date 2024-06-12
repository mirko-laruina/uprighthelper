const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
let net;
var audio = new Audio('audio_file.mp3');


async function app() {

  const localStorageKey = "classifierData";
  const gracePeriod = 10

  var paused = false;


  const loadFromLocalStorage = storageKey => {
    // Load the model.
    var dataset = localStorage.getItem(storageKey);
    if(dataset !== undefined && dataset !== null){
      console.log("Loading dataset from local storage (" + storageKey + ")");
      var objects = Object.fromEntries( JSON.parse(dataset).map(([label, data, shape])=>[label, tf.tensor(data, shape)]) )
      for (const [label, tensor] of Object.entries(objects)){
        // for each row of the tensor, add it to the classifier
        for (let i = 0; i < tensor.shape[0]; i++) {
          classifier.addExample(tensor.slice(i, 1), parseInt(label));
        }
      }

      // This should be quicker, but it doesn't work
      // classifier.setClassifierDataset( Object.fromEntries( JSON.parse(dataset).map(([label, data, shape])=>[label, tf.tensor(data, shape)]) ) );
      console.log("Successfully loaded dataset from local storage (" + storageKey + ")");
    }
  }

  loadFromLocalStorage(localStorageKey);
  // Setup the webcam before loading the model so that the user can grant the webcam permission while the model is loading.
  await setupWebcam();
  console.log('Loading mobilenet..');
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');


  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = classId => {
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(webcamElement, 'conv_preds');

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);
    activation.dispose()

    // persist
    let dataset = JSON.stringify(Object.entries(classifier.getClassifierDataset()).map(([label, data])=>[label, Array.from(data.dataSync()), data.shape]) )
    localStorage.setItem(localStorageKey, dataset)
  };

  const pause = async () => {
    if(paused){
      console.log("Resuming")
      await startWebcam()
      paused = false;
    } else {
      console.log("Pausing")
      paused = true;
      await stopWebcam()
    }    
  }

  const clearStorage = async () => {
    var success = confirm("All data will be lost. Are you sure you want to reset?")
    if(!success){
      return;
    }
    localStorage.clear();
    location.reload();
  }

  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('pause').addEventListener('click', (evt) => {
    pause()
    evt.target.innerHTML = paused ? "Start" : "Pause"
  });
  document.getElementById('reset').addEventListener('click', clearStorage);



  let startBadTime = null;
  
  while (true) {
    await tf.nextFrame()
    if(paused){
      continue;
    }

    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['A', 'B'];
      if(classes[result.classIndex]=="B"){
        if(startBadTime === null){
          startBadTime = new Date()
        }
        if(new Date() - startBadTime < gracePeriod * 1000){
          // grace
        } else {
          await audio.play();
        }
        document.body.style.backgroundColor = "rgb(168, 63, 63)";
      }
      else{
        document.body.style.backgroundColor = "rgb(80, 168, 80)";
        startBadTime = null
      }

      activation.dispose()
    }
  }
}

async function stopWebcam() {
  webcamElement.srcObject.getVideoTracks().forEach(track => track.stop());
}

async function startWebcam() {
  await setupWebcam();
}

async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
      navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
      navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia({ video: true },
        stream => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata', () => resolve(), false);
        },
        error => reject());
    } else {
      reject();
    }
  });
}

app();