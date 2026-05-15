import {useCallback, useState} from 'react'
import {searchSimilarCars} from '../services/api'

export default function SearchPage() {
    const [imageFile, setImageFile] = useState(null)
    const [imagePreview, setImagePreview] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [results, setResults] = useState(null)
    const [dragOver, setDragOver] = useState(false)
    const [plateQuery, setPlateQuery] = useState('')

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

    const handleSearch = async () => {
        if (!imageFile) return

        setLoading(true)
        setError(null)
        try {
            const data = await searchSimilarCars(imageFile, plateQuery)
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
    }

    return (
        <div className="container">
            <div className="card">
                <h1 className="card-title">Search for Similar Cars</h1>
                <p style={{color: 'var(--gray-500)', marginBottom: '1.5rem'}}>
                    Upload a car photo with a partially visible license plate to find the top 5 most similar vehicles.
                </p>

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

                <div className="form-group">
                    <label className="form-label" htmlFor="search-plate-query">
                        Visible plate fragment
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
                        disabled={!imageFile || loading}
                    >
                        {loading ? <span className="spinner"/> : '🔍'}
                        {loading ? 'Searching...' : 'Search Similar Cars'}
                    </button>
                    {imageFile && (
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
              Detected plate: <strong>{results.detected_plate || 'N/A'}</strong>
                            {results.plate_query ? (
                                <> &middot; Query: <strong>{results.plate_query}</strong></>
                            ) : null}
                            {' '}&middot; {results.total_found} car(s) found
            </span>
                    </div>

                    {results.results.length === 0 ? (
                        <p style={{color: 'var(--gray-500)', textAlign: 'center', padding: '2rem'}}>
                            No similar cars found in the database.
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
                    ✓ {(car.similarity_score * 100).toFixed(1)}% similarity
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
