document.addEventListener('DOMContentLoaded', async () => {
  const modelSelect = document.getElementById('modelSelect');
  try {
    const resp = await axios.get('/models');
    resp.data.models.forEach((m) => {
      const opt = document.createElement('option');
      opt.value = m.key;
      opt.textContent = m.description;
      modelSelect.appendChild(opt);
    });
  } catch (err) {
    console.error('Failed to load model registry', err);
  }
});

document.getElementById('uploadForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById('fileInput');
  const modelSelect = document.getElementById('modelSelect');
  const resultDiv = document.getElementById('result');
  const errorDiv = document.getElementById('error');
  const predLabel = document.getElementById('predLabel');
  const predConf = document.getElementById('predConf');
  const submitBtn = document.querySelector('#uploadForm button');

  resultDiv.style.display = 'none';
  errorDiv.style.display = 'none';

  if (!fileInput.files.length) {
    errorDiv.textContent = 'Please choose an image file.';
    errorDiv.style.display = 'block';
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('model_key', modelSelect.value);

  submitBtn.disabled = true;
  submitBtn.textContent = 'Loading...';

  try {
    const resp = await axios.post('/predict', formData);
    const data = resp.data;
    predLabel.textContent = `${data.prediction_label}`;
    predConf.textContent = data.confidence.toFixed(4);
    resultDiv.style.display = 'block';
  } catch (err) {
    const msg = err.response?.data?.error || 'Request failed';
    errorDiv.textContent = msg;
    errorDiv.style.display = 'block';
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Run Inference';
  }
});
