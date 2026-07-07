import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import TopBar from './components/TopBar'
import HomeView from './components/HomeView'
import ChatView from './components/ChatView'

export default function App() {
  return (
    <BrowserRouter>
      <div className="h-screen w-screen flex bg-background text-foreground overflow-hidden">
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <TopBar />
          <Routes>
            <Route path="/" element={<HomeView />} />
            <Route path="/chat/:sessionId" element={<ChatView />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  )
}
