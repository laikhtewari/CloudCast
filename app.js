const express = require('express');
const spawn = require("child_process").spawn;

const app = express();
app.use(express.static('test_images'))

const server = app.listen(7000, () => {
  console.log(`Express running → PORT ${server.address().port}`);
});

app.get('/', (req, res) => {
	const params = req.query;
	const image_id = params.image_id ? params.image_id : 0;
	console.log(`Running inference on image ${image_id}`);
	const pythonProcess = spawn('python3',["inference.py", image_id]);
	pythonProcess.stdout.on('data', (data) => {
		py_res = JSON.parse(data);
		if ('error' in py_res) {
			res.send('Python error: ' + py_res.error)
		} else {
			res.send(render(image_id, py_res.pv_pred, py_res.model_latency, py_res.rt_latency));
		}
	});
});

function pad(n) {
    const s = "000" + n;
    return s.substr(s.length-4);
}

function render(image_id, pred, model_latency, rt_latency) {
	return `<!DOCTYPE html>
			<html>
			<head>
				<meta charset="utf-8">
				<title>PV Output Prediction</title>
			</head>
			<body>
			<div style="display: flex; flex-direction: column; width: 50%; margin: 0 auto; text-align: center;">
				<h1>Predicting PV Output from Cloud Image ${image_id}</h1>
				<img src="/image${pad(image_id)}.png" style="margin: 0 auto; width: 256px">
				<div style="display: flex; justify-content: space-between;">
					<p>Prediction: ${pred} KWH</p>
					<p>Model Latency: ${model_latency} ms</p>
					<p>Round Trip Latency: ${rt_latency} ms</p>
				</div>
				<form action="/" method="get">
					<input type="number" name="image_id" placeholder="Enter image ID">
					<input type="submit" value="Run">
				</form>
			</div>
			</body>
			</html>`
}
