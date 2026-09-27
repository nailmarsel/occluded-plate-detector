const API_BASE = import.meta.env.VITE_API_URL || '';

export async function searchSimilarCars(imageFile, plateQuery = '', description = '') {
    const formData = new FormData();
    formData.append('image', imageFile);
    if (plateQuery.trim()) formData.append('plate_query', plateQuery.trim());
    if (description.trim()) formData.append('description', description.trim());

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

export async function searchByDescription(description, plateQuery = '') {
    const response = await fetch(`${API_BASE}/api/v1/search/text`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            description: description.trim(),
            plate_query: plateQuery.trim(),
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({message: 'Search failed'}));
        const detail = Array.isArray(error.detail) ? error.detail[0]?.msg : error.detail?.message;
        throw new Error(detail || error.message || 'Search failed');
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

export async function batchIndexCars(archiveFile, prefix) {
    const formData = new FormData();
    formData.append('archive', archiveFile);
    if (prefix) formData.append('prefix', prefix);

    const response = await fetch(`${API_BASE}/api/v1/index/batch/zip`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({message: 'Batch indexing failed'}));
        throw new Error(error.detail?.message || error.message || 'Batch indexing failed');
    }

    return response.json();
}
