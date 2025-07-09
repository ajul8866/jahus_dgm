import React, { useState, useEffect, useRef } from 'react';
import { useSocket } from '../services/SocketService';
import apiService from '../services/ApiService';
import './Header.css';

const Header = ({ activeView, currentSession, systemMetrics, setSystemMetrics, addNotification }) => {
  const { connected } = useSocket();
  const [currentTime, setCurrentTime] = useState(new Date());
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [showSearch, setShowSearch] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [systemHealth, setSystemHealth] = useState('healthy');
  
  const searchRef = useRef(null);
  const notificationRef = useRef(null);
  const userMenuRef = useRef(null);

  // View titles mapping
  const viewTitles = {
    dashboard: 'System Dashboard',
    evolution: 'Evolution Control Center',
    agents: 'Agent Management',
    code: 'Code Editor & Generator',
    analytics: 'Performance Analytics',
    llm: 'LLM Integration Hub',
    monitor: 'System Monitor'
  };

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Load system metrics periodically
  useEffect(() => {
    const loadMetrics = async () => {
      try {
        const metrics = await apiService.getSystemMetrics();
        setSystemMetrics(metrics);
        
        // Determine system health
        const health = metrics.systemLoad > 80 ? 'critical' : 
                      metrics.systemLoad > 60 ? 'warning' : 'healthy';
        setSystemHealth(health);
      } catch (error) {
        console.error('Failed to load system metrics:', error);
      }
    };

    loadMetrics();
    const interval = setInterval(loadMetrics, 5000);
    return () => clearInterval(interval);
  }, [setSystemMetrics]);

  // Handle search
  const handleSearch = async (query) => {
    setSearchQuery(query);
    if (query.length > 2) {
      // Simulate search results
      const results = [
        { type: 'agent', id: '1', name: `Agent matching "${query}"`, description: 'High-performance evolved agent' },
        { type: 'session', id: '2', name: `Session containing "${query}"`, description: 'Active evolution session' },
        { type: 'code', id: '3', name: `Code snippet: ${query}`, description: 'Generated code fragment' }
      ];
      setSearchResults(results);
      setShowSearch(true);
    } else {
      setShowSearch(false);
    }
  };

  // Handle click outside to close dropdowns
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setShowSearch(false);
      }
      if (notificationRef.current && !notificationRef.current.contains(event.target)) {
        setShowNotifications(false);
      }
      if (userMenuRef.current && !userMenuRef.current.contains(event.target)) {
        setShowUserMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Mock notifications
  useEffect(() => {
    const mockNotifications = [
      {
        id: 1,
        type: 'success',
        title: 'Evolution Complete',
        message: 'Session #abc123 completed with 95% fitness',
        time: new Date(Date.now() - 300000),
        read: false
      },
      {
        id: 2,
        type: 'warning',
        title: 'High System Load',
        message: 'CPU usage at 78%, consider optimizing',
        time: new Date(Date.now() - 600000),
        read: false
      },
      {
        id: 3,
        type: 'info',
        title: 'New Agent Generated',
        message: 'Agent #xyz789 shows promising results',
        time: new Date(Date.now() - 900000),
        read: true
      }
    ];
    setNotifications(mockNotifications);
  }, []);

  const formatTime = (date) => {
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const formatDate = (date) => {
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const getRelativeTime = (date) => {
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return date.toLocaleDateString();
  };

  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <header className="header">
      <div className="header-left">
        <div className="breadcrumb">
          <div className="breadcrumb-item">
            <i className="fas fa-home"></i>
            <span>DGM</span>
          </div>
          <div className="breadcrumb-separator">
            <i className="fas fa-chevron-right"></i>
          </div>
          <div className="breadcrumb-item active">
            <span>{viewTitles[activeView] || 'Unknown View'}</span>
          </div>
        </div>
        
        {currentSession && (
          <div className="session-indicator">
            <div className={`session-status ${currentSession.status}`}>
              <div className="status-dot"></div>
            </div>
            <span className="session-name">
              Session: {currentSession.id.substring(0, 8)}...
            </span>
            {currentSession.currentGeneration !== undefined && (
              <span className="generation-info">
                Gen {currentSession.currentGeneration}
              </span>
            )}
          </div>
        )}
      </div>

      <div className="header-center">
        <div className="search-container" ref={searchRef}>
          <div className="search-input-wrapper">
            <i className="fas fa-search search-icon"></i>
            <input
              type="text"
              className="search-input"
              placeholder="Search agents, sessions, code..."
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
              onFocus={() => searchQuery.length > 2 && setShowSearch(true)}
            />
            {searchQuery && (
              <button 
                className="search-clear"
                onClick={() => {
                  setSearchQuery('');
                  setShowSearch(false);
                }}
              >
                <i className="fas fa-times"></i>
              </button>
            )}
          </div>
          
          {showSearch && searchResults.length > 0 && (
            <div className="search-dropdown">
              <div className="search-results">
                {searchResults.map(result => (
                  <div key={result.id} className="search-result-item">
                    <div className="result-icon">
                      <i className={`fas ${
                        result.type === 'agent' ? 'fa-robot' :
                        result.type === 'session' ? 'fa-flask' :
                        'fa-code'
                      }`}></i>
                    </div>
                    <div className="result-content">
                      <div className="result-name">{result.name}</div>
                      <div className="result-description">{result.description}</div>
                    </div>
                    <div className="result-type">{result.type}</div>
                  </div>
                ))}
              </div>
              <div className="search-footer">
                <span>Press Enter to search all results</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="header-right">
        {/* System Health Indicator */}
        <div className={`health-indicator ${systemHealth}`} title="System Health">
          <i className={`fas ${
            systemHealth === 'healthy' ? 'fa-heart' :
            systemHealth === 'warning' ? 'fa-exclamation-triangle' :
            'fa-exclamation-circle'
          }`}></i>
        </div>

        {/* Connection Status */}
        <div className={`connection-indicator ${connected ? 'connected' : 'disconnected'}`}>
          <div className="connection-dot"></div>
          <span className="connection-text">
            {connected ? 'Online' : 'Offline'}
          </span>
        </div>

        {/* System Metrics */}
        <div className="metrics-display">
          <div className="metric-item" title="CPU Usage">
            <i className="fas fa-microchip"></i>
            <span>{systemMetrics.systemLoad ? Math.round(systemMetrics.systemLoad) : 0}%</span>
          </div>
          <div className="metric-item" title="Active Agents">
            <i className="fas fa-robot"></i>
            <span>{systemMetrics.totalAgents || 0}</span>
          </div>
          <div className="metric-item" title="Best Fitness">
            <i className="fas fa-trophy"></i>
            <span>{systemMetrics.bestFitness ? (systemMetrics.bestFitness * 100).toFixed(0) + '%' : '0%'}</span>
          </div>
        </div>

        {/* Notifications */}
        <div className="notification-container" ref={notificationRef}>
          <button 
            className="notification-btn"
            onClick={() => setShowNotifications(!showNotifications)}
          >
            <i className="fas fa-bell"></i>
            {unreadCount > 0 && (
              <span className="notification-badge">{unreadCount}</span>
            )}
          </button>
          
          {showNotifications && (
            <div className="notification-dropdown">
              <div className="notification-header">
                <h3>Notifications</h3>
                <button className="mark-all-read">Mark all read</button>
              </div>
              
              <div className="notification-list">
                {notifications.map(notification => (
                  <div 
                    key={notification.id} 
                    className={`notification-item ${notification.read ? 'read' : 'unread'}`}
                  >
                    <div className={`notification-icon ${notification.type}`}>
                      <i className={`fas ${
                        notification.type === 'success' ? 'fa-check-circle' :
                        notification.type === 'warning' ? 'fa-exclamation-triangle' :
                        notification.type === 'error' ? 'fa-times-circle' :
                        'fa-info-circle'
                      }`}></i>
                    </div>
                    <div className="notification-content">
                      <div className="notification-title">{notification.title}</div>
                      <div className="notification-message">{notification.message}</div>
                      <div className="notification-time">{getRelativeTime(notification.time)}</div>
                    </div>
                    {!notification.read && <div className="unread-dot"></div>}
                  </div>
                ))}
              </div>
              
              <div className="notification-footer">
                <button className="view-all-btn">View All Notifications</button>
              </div>
            </div>
          )}
        </div>

        {/* Time Display */}
        <div className="time-display">
          <div className="time">{formatTime(currentTime)}</div>
          <div className="date">{formatDate(currentTime)}</div>
        </div>

        {/* User Menu */}
        <div className="user-menu-container" ref={userMenuRef}>
          <button 
            className="user-menu-btn"
            onClick={() => setShowUserMenu(!showUserMenu)}
          >
            <div className="user-avatar">
              <i className="fas fa-user"></i>
            </div>
            <div className="user-info">
              <div className="user-name">Admin</div>
              <div className="user-role">System Administrator</div>
            </div>
            <i className="fas fa-chevron-down"></i>
          </button>
          
          {showUserMenu && (
            <div className="user-dropdown">
              <div className="user-dropdown-header">
                <div className="user-avatar large">
                  <i className="fas fa-user"></i>
                </div>
                <div className="user-details">
                  <div className="user-name">Administrator</div>
                  <div className="user-email">admin@dgm.local</div>
                </div>
              </div>
              
              <div className="user-menu-items">
                <button className="user-menu-item">
                  <i className="fas fa-user-cog"></i>
                  <span>Profile Settings</span>
                </button>
                <button className="user-menu-item">
                  <i className="fas fa-cog"></i>
                  <span>System Settings</span>
                </button>
                <button className="user-menu-item">
                  <i className="fas fa-download"></i>
                  <span>Export Data</span>
                </button>
                <button className="user-menu-item">
                  <i className="fas fa-question-circle"></i>
                  <span>Help & Support</span>
                </button>
                <div className="menu-separator"></div>
                <button className="user-menu-item danger">
                  <i className="fas fa-sign-out-alt"></i>
                  <span>Sign Out</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;