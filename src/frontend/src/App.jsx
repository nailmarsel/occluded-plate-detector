import {NavLink, Route, Routes} from 'react-router-dom'
import SearchPage from './components/SearchPage'
import IndexPage from './components/IndexPage'

export default function App() {
    return (
        <>
            <header className="header">
                <div className="header-inner">
                    <div className="logo">
                        🚗 Autobahn<span>CV</span>
                    </div>
                    <nav className="nav">
                        <NavLink to="/" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
                            Search
                        </NavLink>
                        <NavLink to="/index" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>
                            Index
                        </NavLink>
                    </nav>
                </div>
            </header>

            <Routes>
                <Route path="/" element={<SearchPage/>}/>
                <Route path="/index" element={<IndexPage/>}/>
            </Routes>
        </>
    )
}
