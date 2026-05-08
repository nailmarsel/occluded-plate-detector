import { useState, useEffect } from 'react'
import { getHealth } from '../services/api'

export default function HealthPage() {
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchHealth = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getHealth()
      setHealth(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHealth()
  }, [])

  const StatusIcon = ({ ok }) => (
    <span style={{ fontSize: '1.25rem' }}>{ok ? '✅' : '❌'}</span>
  )

  return (
    <div className="container">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
        <h1>System Health</h1>
        <button className="btn btn-secondary" onClick={fetchHealth} disabled={loading}>
          {loading ? <span className="spinner" /> : '🔄'} Refresh
        </button>
      </div>

      {error && <div className="alert alert-error"><strong>Error:</strong> {error}</div>}

      {loading && !health && (
        <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
          <span className="spinner" style={{ width: 40, height: 40 }} />
        </div>
      )}

      {health && (
        <>
          <div className="card">
            <h2 className="card-title">Overall Status</h2>
            <span className={`badge ${
              health.status === 'healthy' ? 'badge-success' :
              health.status === 'degraded' ? 'badge-warning' : 'badge-error'
            }`}>
              {health.status}
            </span>
          </div>

          <div className="card">
            <h2 className="card-title">Services</h2>
            <div className="health-grid">
              <div className="health-item">
                <span className="health-label">Elasticsearch</span>
                <StatusIcon ok={health.elasticsearch} />
              </div>
              {health.neurons && Object.entries(health.neurons).map(([name, status]) => (
                <div key={name} className="health-item">
                  <span className="health-label">{name.replace(/_/g, ' ')}</span>
                  <StatusIcon ok={status === 'initialized'} />
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
