import {useCallback, useState} from 'react'
import {searchByDescription, searchSimilarCars} from '../services/api'

const MODE_IMAGE = 'image'
const MODE_TEXT = 'text'

const DESCRIPTION_EXAMPLES = [
    'белая Toyota Camry седан, номер А1**ВС77',
    'чёрный внедорожник с багажником на крыше',
    'red hatchback, plate ?123??77',
]

export default function SearchPage() {
    const [imageFile, setImageFile] = useState(null)
    const [imagePreview, setImagePreview] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [results, setResults] = useState(null)
    const [dragOver, setDragOver] = useState(false)
    const [plateQuery, setPlateQuery] = useState('')
    const [mode, setMode] = useState(MODE_IMAGE)
    const [description, setDescription] = useState('')

    const handleFileSelect = useCallback((file) => {
        if (file && (file.type === 'image/jpeg' || file.type === 'image/png' || file.type === 'image/jpg')) {
            setImageFile(file)
            setImagePreview(URL.createObjectURL(file))
            setError(null)
            setResults(null)
        } else {
            setError('Please select a valid JPEG or PNG image file.')
        }
    }, [])

    const handleDrop = useCallback((e) => {
        e.preventDefault()
        setDragOver(false)
        const file = e.dataTransfer.files[0]
        handleFileSelect(file)
    }, [handleFileSelect])

    const handleDragOver = useCallback((e) => {
        e.preventDefault()
        setDragOver(true)
    }, [])

    const handleDragLeave = useCallback(() => {
        setDragOver(false)
    }, [])

    const canSearch = mode === MODE_IMAGE
        ? Boolean(imageFile)
        : Boolean(description.trim() || plateQuery.trim())

    const handleSearch = async () => {
        if (!canSearch) return

        setLoading(true)
        setError(null)
        try {
            const data = mode === MODE_IMAGE
                ? await searchSimilarCars(imageFile, plateQuery, description)
                : await searchByDescription(description, plateQuery)
            setResults(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const handleReset = () => {
        setImageFile(null)
        setImagePreview(null)
        setResults(null)
        setError(null)
        setPlateQuery('')
        setDescription('')
    }

    const handleModeChange = (nextMode) => {
        if (nextMode === mode) return
        setMode(nextMode)
        setResults(null)
        setError(null)
    }

    const isPlateOnly = results?.search_mode === 'plate'

    return (
        <div className="container">
            <div className="card">
                <h1 className="card-title">Search for Similar Cars</h1>

                <div className="mode-switch" role="tablist" aria-label="Search mode">
                    <button
                        type="button"
                        role="tab"
                        aria-selected={mode === MODE_IMAGE}
                        className={`mode-switch-btn ${mode === MODE_IMAGE ? 'active' : ''}`}
                        onClick={() => handleModeChange(MODE_IMAGE)}
                        disabled={loading}
                    >
                        📷 By photo
                    </button>
                    <button
                        type="button"
                        role="tab"
                        aria-selected={mode === MODE_TEXT}
                        className={`mode-switch-btn ${mode === MODE_TEXT ? 'active' : ''}`}
                        onClick={() => handleModeChange(MODE_TEXT)}
                        disabled={loading}
                    >
                        📝 By description
                    </button>
                </div>

                <p style={{color: 'var(--gray-500)', marginBottom: '1.5rem'}}>
                    {mode === MODE_IMAGE
                        ? 'Upload a car photo with a partially visible license plate to find the top 5 most similar vehicles.'
                        : 'Describe the car in Russian or English — color, make, body type, details — and optionally a plate fragment. No photo needed.'}
                </p>

                {mode === MODE_IMAGE && (<>
                <div
                    className={`upload-zone ${dragOver ? 'dragover' : ''}`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onClick={() => document.getElementById('search-file').click()}
                >
                    <div className="upload-zone-icon">📷</div>
                    <div className="upload-zone-text">
                        {imageFile ? imageFile.name : 'Drop an image here or click to upload'}
                    </div>
                    <div className="upload-zone-hint">JPEG, PNG supported</div>
                    <input
                        id="search-file"
                        type="file"
                        accept="image/jpeg,image/png"
                        style={{display: 'none'}}
                        onChange={(e) => handleFileSelect(e.target.files[0])}
                    />
                </div>

                {imagePreview && (
                    <div className="image-preview">
                        <img src={imagePreview} alt="Uploaded car"/>
                    </div>
                )}
                </>)}

                <div className="form-group">
                    <label className="form-label" htmlFor="search-description">
                        {mode === MODE_IMAGE ? 'Description (optional)' : 'Car description'}
                    </label>
                    <textarea
                        id="search-description"
                        className="form-input"
                        rows={mode === MODE_IMAGE ? 2 : 3}
                        maxLength={500}
                        placeholder={mode === MODE_IMAGE
                            ? 'белый седан, литые диски'
                            : DESCRIPTION_EXAMPLES[0]}
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        onKeyDown={(e) => {
                            if (mode === MODE_TEXT && e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleSearch()
                        }}
                        disabled={loading}
                    />
                    {mode === MODE_TEXT && (
                        <div className="example-chips">
                            {DESCRIPTION_EXAMPLES.map((example) => (
                                <button
                                    key={example}
                                    type="button"
                                    className="example-chip"
                                    onClick={() => setDescription(example)}
                                    disabled={loading}
                                >
                                    {example}
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                <div className="form-group">
                    <label className="form-label" htmlFor="search-plate-query">
                        Visible plate fragment{mode === MODE_TEXT ? ' (optional, overrides one in the description)' : ''}
                    </label>
                    <input
                        id="search-plate-query"
                        className="form-input"
                        type="text"
                        placeholder="A8**AA977"
                        value={plateQuery}
                        onChange={(e) => setPlateQuery(e.target.value)}
                        disabled={loading}
                    />
                </div>

                <div style={{display: 'flex', gap: '0.75rem', marginTop: '1rem'}}>
                    <button
                        className="btn btn-primary btn-block"
                        onClick={handleSearch}
                        disabled={!canSearch || loading}
                    >
                        {loading ? <span className="spinner"/> : '🔍'}
                        {loading ? 'Searching...' : (mode === MODE_IMAGE ? 'Search Similar Cars' : 'Search by Description')}
                    </button>
                    {(imageFile || description || plateQuery) && (
                        <button className="btn btn-secondary" onClick={handleReset} disabled={loading}>
                            Reset
                        </button>
                    )}
                </div>
            </div>

            {error && (
                <div className="alert alert-error">
                    <strong>Error:</strong> {error}
                </div>
            )}

            {results && (
                <div className="card">
                    <div className="results-header">
                        <h2 className="card-title" style={{marginBottom: 0}}>
                            Search Results
                        </h2>
                        <span className="results-count">
                            {results.search_mode === 'image' || results.search_mode === 'image+text' ? (
                                <>Detected plate: <strong>{results.detected_plate || 'N/A'}</strong></>
                            ) : (
                                <>Description: <strong>{results.text_query || '—'}</strong></>
                            )}
                            {results.plate_query ? (
                                <> &middot; Plate: <strong>{results.plate_query}</strong></>
                            ) : null}
                            {' '}&middot; {results.total_found} car(s) found
                        </span>
                    </div>

                    {results.results.length === 0 ? (
                        <p style={{color: 'var(--gray-500)', textAlign: 'center', padding: '2rem'}}>
                            {mode === MODE_TEXT
                                ? 'No cars match this description. Try fewer details or a shorter plate fragment.'
                                : 'No similar cars found in the database.'}
                        </p>
                    ) : (
                        results.results.map((car, index) => (
                            <div key={index} className="result-card">
                                {car.image_url ? (
                                    <img className="result-image" src={car.image_url} alt={`Car ${car.car_id}`}/>
                                ) : (
                                    <div className="result-image" style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        color: 'var(--gray-400)'
                                    }}>
                                        No image
                                    </div>
                                )}
                                <div className="result-info">
                                    <div className="result-plate">{car.plate_number}</div>
                                    <div className="result-id">ID: {car.car_id}</div>
                                    <span className="result-score">
                                        {isPlateOnly
                                            ? '✓ plate match'
                                            : `✓ ${(car.similarity_score * 100).toFixed(1)}% ${results.search_mode === 'text' ? 'relevance' : 'similarity'}`}
                                    </span>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            )}
        </div>
    )
}
