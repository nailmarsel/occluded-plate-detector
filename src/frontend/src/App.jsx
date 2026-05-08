import { Routes, Route, NavLink } from 'react-router-dom'
import SearchPage from './components/SearchPage'
import IndexPage from './components/IndexPage'
import HealthPage from './components/HealthPage'

export default function App() {
  return (
    <>
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            🚗 Autobahn<span>CV</span>
          </div>
          <nav className="nav">
            <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              Search
            </NavLink>
            <NavLink to="/index" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              Index
            </NavLink>
            <NavLink to="/health" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              Health
            </NavLink>
          </nav>
        </div>
      </header>

      <Routes>
        <Route path="/" element={<SearchPage />} />
        <Route path="/index" element={<IndexPage />} />
        <Route path="/health" element={<HealthPage />} />
      </Routes>
    </>
  )
}
