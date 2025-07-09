import React, { useState, useEffect } from 'react';
import { SocketProvider } from './services/SocketService';
import Dashboard from './components/Dashboard';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import EvolutionControl from './components/EvolutionControl';
import AgentViewer from './components/AgentViewer';
import CodeEditor from './components/CodeEditor';
import AnalyticsPanel from './components/AnalyticsPanel';
import LLMInterface from './components/LLMInterface';
import SystemMonitor from './components/SystemMonitor';
import './styles/App.css';

const App = () => {
  const [activeView, setActiveView] = useState('dashboard');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [currentSession, setCurrentSession] = useState(null);
  const [systemMetrics, setSystemMetrics] = useState({});
  const [notifications, setNotifications] = useState([]);

  const views = {
    dashboard: <Dashboard currentSession={currentSession} />,
    evolution: <EvolutionControl currentSession={currentSession} setCurrentSession={setCurrentSession} />,
    agents: <AgentViewer currentSession={currentSession} />,
    code: <CodeEditor currentSession={currentSession} />,
    analytics: <AnalyticsPanel currentSession={currentSession} />,
    llm: <LLMInterface currentSession={currentSession} />,
    monitor: <SystemMonitor systemMetrics={systemMetrics} />
  };

  const addNotification = (notification) => {
    const id = Date.now();
    setNotifications(prev => [...prev, { ...notification, id }]);
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 5000);
  };

  return (
    <SocketProvider>
      <div className="app">
        <Sidebar 
          activeView={activeView}
          setActiveView={setActiveView}
          collapsed={sidebarCollapsed}
          setCollapsed={setSidebarCollapsed}
          currentSession={currentSession}
        />
        
        <div className={`main-content ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
          <Header 
            activeView={activeView}
            currentSession={currentSession}
            systemMetrics={systemMetrics}
            setSystemMetrics={setSystemMetrics}
            addNotification={addNotification}
          />
          
          <div className="content-area">
            {views[activeView]}
          </div>
        </div>
        
        {/* Notifications */}
        <div className="notifications-container">
          {notifications.map(notification => (
            <div key={notification.id} className={`notification ${notification.type}`}>
              <i className={`fas ${notification.icon || 'fa-info-circle'}`}></i>
              <div className="notification-content">
                <div className="notification-title">{notification.title}</div>
                <div className="notification-message">{notification.message}</div>
              </div>
              <button 
                className="notification-close"
                onClick={() => setNotifications(prev => prev.filter(n => n.id !== notification.id))}
              >
                <i className="fas fa-times"></i>
              </button>
            </div>
          ))}
        </div>
      </div>
    </SocketProvider>
  );
};

export default App;