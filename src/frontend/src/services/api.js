const API_BASE = import.meta.env.VITE_API_URL || '';

export async function searchSimilarCars(imageFile) {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await fetch(`${API_BASE}/api/v1/search`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({message: 'Search failed'}));
        throw new Error(error.detail?.message || error.message || 'Search failed');
    }

    return response.json();
}

export async function indexCar(imageFile, plateNumber) {
    const formData = new FormData();
    formData.append('image', imageFile);
    if (plateNumber) formData.append('plate_number', plateNumber);

    const response = await fetch(`${API_BASE}/api/v1/index`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({message: 'Indexing failed'}));
        throw new Error(error.detail?.message || error.message || 'Indexing failed');
    }

    return response.json();
}

export async function batchIndexCars(folderPath, prefix) {
    const response = await fetch(`${API_BASE}/api/v1/index/batch`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({folder_path: folderPath, prefix: prefix || null}),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({message: 'Batch indexing failed'}));
        throw new Error(error.detail?.message || error.message || 'Batch indexing failed');
    }

    return response.json();
}
