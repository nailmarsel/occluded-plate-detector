import {useCallback, useState} from 'react'
import {batchIndexCars, indexCar} from '../services/api'

export default function IndexPage() {
    const [activeTab, setActiveTab] = useState('single')

    return (
        <div className="container">
            <h1 style={{marginBottom: '1.5rem'}}>Index Cars</h1>

            <div className="tabs">
                <button
                    className={`tab ${activeTab === 'single' ? 'active' : ''}`}
                    onClick={() => setActiveTab('single')}
                >
                    Single Car
                </button>
                <button
                    className={`tab ${activeTab === 'batch' ? 'active' : ''}`}
                    onClick={() => setActiveTab('batch')}
                >
                    Batch from Folder
                </button>
            </div>

            {activeTab === 'single' ? <SingleIndexForm/> : <BatchIndexForm/>}
        </div>
    )
}

function SingleIndexForm() {
    const [imageFile, setImageFile] = useState(null)
    const [imagePreview, setImagePreview] = useState(null)
    const [plateNumber, setPlateNumber] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [success, setSuccess] = useState(null)
    const [dragOver, setDragOver] = useState(false)

    const handleFileSelect = useCallback((file) => {
        if (file && (file.type === 'image/jpeg' || file.type === 'image/png')) {
            setImageFile(file)
            setImagePreview(URL.createObjectURL(file))
            setError(null)
            setSuccess(null)
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

    const handleSubmit = async () => {
        if (!imageFile) return

        setLoading(true)
        setError(null)
        try {
            const data = await indexCar(imageFile, plateNumber.trim())
            setSuccess(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    const handleReset = () => {
        setImageFile(null)
        setImagePreview(null)
        setPlateNumber('')
        setSuccess(null)
        setError(null)
    }

    return (
        <>
            <div className="card">
                <h2 className="card-title">Index a Single Car</h2>

                <div
                    className={`upload-zone ${dragOver ? 'dragover' : ''}`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onClick={() => document.getElementById('index-file').click()}
                >
                    <div className="upload-zone-icon">📷</div>
                    <div className="upload-zone-text">
                        {imageFile ? imageFile.name : 'Drop an image here or click to upload'}
                    </div>
                    <input
                        id="index-file"
                        type="file"
                        accept="image/jpeg,image/png"
                        style={{display: 'none'}}
                        onChange={(e) => handleFileSelect(e.target.files[0])}
                    />
                </div>

                {imagePreview && (
                    <div className="image-preview">
                        <img src={imagePreview} alt="Car to index"/>
                    </div>
                )}

                <div className="form-group">
                    <label className="form-label" htmlFor="index-plate">
                        Plate Number
                    </label>
                    <input
                        id="index-plate"
                        className="form-input"
                        type="text"
                        placeholder="A864AA199"
                        value={plateNumber}
                        onChange={(e) => setPlateNumber(e.target.value)}
                    />
                </div>

                <div style={{display: 'flex', gap: '0.75rem', marginTop: '1rem'}}>
                    <button
                        className="btn btn-primary btn-block"
                        onClick={handleSubmit}
                        disabled={!imageFile || loading}
                    >
                        {loading ? <span className="spinner"/> : '📤'}
                        {loading ? 'Processing...' : 'Index Car'}
                    </button>
                    {imageFile && (
                        <button className="btn btn-secondary" onClick={handleReset} disabled={loading}>
                            Reset
                        </button>
                    )}
                </div>
            </div>

            {error && <div className="alert alert-error"><strong>Error:</strong> {error}</div>}

            {success && (
                <div className="alert alert-success">
                    <strong>Success!</strong> Car indexed: {success.car_id}, plate: {success.plate_number}, embedding
                    dim: {success.embedding_dim}
                </div>
            )}
        </>
    )
}

function BatchIndexForm() {
    const [folderPath, setFolderPath] = useState('')
    const [prefix, setPrefix] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [results, setResults] = useState(null)

    const handleSubmit = async () => {
        if (!folderPath.trim()) return

        setLoading(true)
        setError(null)
        try {
            const data = await batchIndexCars(folderPath.trim(), prefix.trim() || null)
            setResults(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <>
            <div className="card">
                <h2 className="card-title">Batch Index from Folder</h2>
                <p style={{color: 'var(--gray-500)', marginBottom: '1rem'}}>
                    Enter the absolute path to a folder containing car photos. All JPEG and PNG files will be processed.
                </p>

                <div className="form-group">
                    <label className="form-label" htmlFor="folder-path">Folder Path</label>
                    <input
                        id="folder-path"
                        className="form-input"
                        type="text"
                        placeholder="/path/to/car/photos"
                        value={folderPath}
                        onChange={(e) => setFolderPath(e.target.value)}
                    />
                </div>

                <div className="form-group">
                    <label className="form-label" htmlFor="prefix">
                        ID Prefix <span style={{color: 'var(--gray-400)'}}>(optional)</span>
                    </label>
                    <input
                        id="prefix"
                        className="form-input"
                        type="text"
                        placeholder="e.g. batch_001"
                        value={prefix}
                        onChange={(e) => setPrefix(e.target.value)}
                    />
                </div>

                <button
                    className="btn btn-primary btn-block"
                    onClick={handleSubmit}
                    disabled={!folderPath.trim() || loading}
                >
                    {loading ? <span className="spinner"/> : '📦'}
                    {loading ? 'Processing...' : 'Start Batch Index'}
                </button>
            </div>

            {error && <div className="alert alert-error"><strong>Error:</strong> {error}</div>}

            {results && (
                <div className="card">
                    <h2 className="card-title">Batch Results</h2>

                    <div className="batch-summary">
                        <div className="batch-stat" style={{background: 'var(--gray-100)'}}>
                            <div className="batch-stat-value">{results.total}</div>
                            <div className="batch-stat-label">Total</div>
                        </div>
                        <div className="batch-stat" style={{background: 'var(--success-light)'}}>
                            <div className="batch-stat-value"
                                 style={{color: 'var(--success)'}}>{results.succeeded}</div>
                            <div className="batch-stat-label">Succeeded</div>
                        </div>
                        <div className="batch-stat" style={{background: 'var(--error-light)'}}>
                            <div className="batch-stat-value" style={{color: 'var(--error)'}}>{results.failed}</div>
                            <div className="batch-stat-label">Failed</div>
                        </div>
                    </div>

                    {results.results.length > 0 && (
                        <div style={{overflowX: 'auto'}}>
                            <table className="batch-table">
                                <thead>
                                <tr>
                                    <th>File</th>
                                    <th>Car ID</th>
                                    <th>Status</th>
                                    <th>Error</th>
                                </tr>
                                </thead>
                                <tbody>
                                {results.results.map((r, i) => (
                                    <tr key={i}>
                                        <td>{r.filename}</td>
                                        <td>{r.car_id}</td>
                                        <td>
                        <span className={`badge ${r.status === 'indexed' ? 'badge-success' : 'badge-error'}`}>
                          {r.status}
                        </span>
                                        </td>
                                        <td style={{color: 'var(--gray-500)', fontSize: '0.75rem'}}>
                                            {r.error || '—'}
                                        </td>
                                    </tr>
                                ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            )}
        </>
    )
}
