let selectedFile = null;

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const resultsGrid = document.getElementById('resultsGrid');

uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

function handleFile(file) {
    if (!file.type.startsWith('image/')) return alert('Выберите изображение');
    if (file.size > 10*1024*1024) return alert('Файл >10MB');
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewSection.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

function clearPreview() {
    selectedFile = null;
    previewSection.style.display = 'none';
    uploadArea.style.display = 'block';
    fileInput.value = '';
    results.style.display = 'none';
}

async function uploadImage() {
    if (!selectedFile) return alert('Выберите файл');
    previewSection.style.display = 'none';
    loading.style.display = 'block';
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
        const response = await fetch('http://localhost:5000/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        loading.style.display = 'none';
        if (data.success) {
            showResults(data.matched_cars);
        } else {
            alert('Ошибка: ' + data.error);
            clearPreview();
        }
    } catch (err) {
        loading.style.display = 'none';
        alert('Ошибка соединения: ' + err.message);
        clearPreview();
    }
}

function showResults(cars) {
    resultsGrid.innerHTML = '';
    if (!cars || cars.length === 0) {
        resultsGrid.innerHTML = '<p>Совпадений не найдено</p>';
        results.style.display = 'block';
        return;
    }
    cars.forEach((car, idx) => {
        const card = document.createElement('div');
        card.className = 'car-card';
        let photoHtml = '';
        if (car.photo_base64) {
            photoHtml = `<img src="data:image/jpeg;base64,${car.photo_base64}" class="car-image">`;
        } else {
            photoHtml = `<div class="no-photo">🚗 Фото отсутствует</div>`;
        }
        card.innerHTML = `
            <div class="car-header">
                <span>#${idx+1}</span>
                <span>ID: ${car.id}</span>
            </div>
            ${photoHtml}
            <div class="car-info">
                <div class="plate">${car.plate_number || '—'}</div>
                <div class="path">${car.photo_path || ''}</div>
            </div>
        `;
        resultsGrid.appendChild(card);
    });
    results.style.display = 'block';
}