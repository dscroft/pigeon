let model;
async function loadModel() {
	console.log("model loading..");
	modelName = "mobilenet";
	model = undefined;
	model = await tf.loadLayersModel('./output/mobilenet/model.json');
	console.log("model loaded..");
}

async function loadFile() {
	console.log("image is in loadfile..");
	document.getElementById("select-file-box").style.display = "table-cell";
  	var fileInputElement = document.getElementById("select-file-image");
  	console.log(fileInputElement.files[0]);
    renderImage(fileInputElement.files[0]);
}

function renderImage(file) {
  var reader = new FileReader();
  console.log("image is here..");
  reader.onload = function(event) {
    img_url = event.target.result;
    console.log("image is here2..");
    document.getElementById("test-image").src = img_url;
  }
  reader.readAsDataURL(file);
}

async function predButton() {
	console.log("model loading..");

	if (model == undefined) {
		alert("Please load the model first..")
	}
	//if (document.getElementById("predict-box").style.display == "none") {
	//	alert("Please load an image first..")
	//}
	console.log(model);
	let image  = document.getElementById("test-image");
	let tensor = preprocessImage(image, modelName);

	let predictions = await model.predict(tensor).data();
	let results = Array.from(predictions)
		.map(function (p, i) {
			return {
				probability: p,
				className: IMAGENET_CLASSES[i]
			};
		}).sort(function (a, b) {
			return b.probability - a.probability;
		}).slice(0, 5);

	console.log(">>", results);
	results = results.sort( function(a, b) {
		return b.probability - a.probability;
	} );
	console.log(">>", results);

	document.getElementById("prediction").innerHTML = results[0].className;
}

function preprocessImage(image, modelName) {
	let tensor = tf.browser.fromPixels(image)
		.resizeNearestNeighbor([224, 224])
		.toFloat();

	if (modelName === undefined) {
		return tensor.expandDims();
	} else if (modelName === "mobilenet") {
		let offset = tf.scalar(127.5);
		return tensor.sub(offset)
			.div(offset)
			.expandDims();
	} else {
		alert("Unknown model name..")
	}
}

