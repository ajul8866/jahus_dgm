import React, { useState, useEffect } from 'react';
import { useSocket } from '../services/SocketService';
import './Sidebar.css';

const Sidebar = ({ activeView, setActiveView, collapsed, setCollapsed, currentSession }) => {
  const { connected, systemMetrics } = useSocket();
  const [sessionStatus, setSessionStatus] = useState('idle');

  const menuItems = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: 'fa-tachometer-alt',
      description: 'System overview and metrics'
    },
    {
      id: 'evolution',
      label: 'Evolution Control',
      icon: 'fa-dna',
      description: 'Manage evolution processes',
      badge: currentSession?.status === 'running' ? 'ACTIVE' : null
    },
    {
      id: 'agents',
      label: 'Agent Viewer',
      icon: 'fa-robot',
      description: 'Browse and analyze agents',
      badge: systemMetrics.totalAgents || 0
    },
    {
      id: 'code',
      label: 'Code Editor',
      icon: 'fa-code',
      description: 'Edit and generate code'
    },
    {
      id: 'analytics',
      label: 'Analytics',
      icon: 'fa-chart-line',
      description: 'Performance analytics'
    },
    {
      id: 'llm',
      label: 'LLM Interface',
      icon: 'fa-brain',
      description: 'AI-powered assistance'
    },
    {
      id: 'monitor',
      label: 'System Monitor',
      icon: 'fa-server',
      description: 'System health monitoring'
    }
  ];

  useEffect(() => {
    if (currentSession) {
      setSessionStatus(currentSession.status || 'idle');
    }
  }, [currentSession]);

  const handleMenuClick = (itemId) => {
    setActiveView(itemId);
    
    // Auto-collapse on mobile
    if (window.innerWidth <= 768) {
      setCollapsed(true);
    }
  };

  return (
    <div className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      {/* Header */}
      <div className="sidebar-header">
        <div className="logo-container">
          <div className="logo-icon">
            <i className="fas fa-dna"></i>
          </div>
          {!collapsed && (
            <div className="logo-text">
              <div className="logo-title">DGM</div>
              <div className="logo-subtitle">Advanced Interface</div>
            </div>
          )}
        </div>
        
        <button 
          className="collapse-btn"
          onClick={() => setCollapsed(!collapsed)}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <i className={`fas ${collapsed ? 'fa-chevron-right' : 'fa-chevron-left'}`}></i>
        </button>
      </div>

      {/* Connection Status */}
      <div className="connection-status">
        <div className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}>
          <div className="status-dot"></div>
          {!collapsed && (
            <span className="status-text">
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          )}
        </div>
      </div>

      {/* Session Info */}
      {currentSession && !collapsed && (
        <div className="session-info">
          <div className="session-header">
            <i className="fas fa-flask"></i>
            <span>Current Session</span>
          </div>
          <div className="session-details">
            <div className="session-id">{currentSession.id.substring(0, 8)}...</div>
            <div className={`session-status ${sessionStatus}`}>
              <div className="status-dot"></div>
              {sessionStatus.toUpperCase()}
            </div>
            {currentSession.currentGeneration !== undefined && (
              <div className="session-generation">
                Gen: {currentSession.currentGeneration}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Navigation Menu */}
      <nav className="sidebar-nav">
        {menuItems.map(item => (
          <div
            key={item.id}
            className={`nav-item ${activeView === item.id ? 'active' : ''}`}
            onClick={() => handleMenuClick(item.id)}
            title={collapsed ? item.description : ''}
          >
            <div className="nav-icon">
              <i className={`fas ${item.icon}`}></i>
            </div>
            
            {!collapsed && (
              <>
                <div className="nav-content">
                  <div className="nav-label">{item.label}</div>
                  <div className="nav-description">{item.description}</div>
                </div>
                
                {item.badge && (
                  <div className="nav-badge">
                    {item.badge}
                  </div>
                )}
              </>
            )}
            
            <div className="nav-indicator"></div>
          </div>
        ))}
      </nav>

      {/* System Stats */}
      {!collapsed && systemMetrics && (
        <div className="sidebar-stats">
          <div className="stats-header">
            <i className="fas fa-chart-bar"></i>
            <span>System Stats</span>
          </div>
          
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-value">{systemMetrics.totalAgents || 0}</div>
              <div className="stat-label">Agents</div>
            </div>
            
            <div className="stat-item">
              <div className="stat-value">{systemMetrics.totalEvolutions || 0}</div>
              <div className="stat-label">Sessions</div>
            </div>
            
            <div className="stat-item">
              <div className="stat-value">
                {systemMetrics.bestFitness ? (systemMetrics.bestFitness * 100).toFixed(1) + '%' : '0%'}
              </div>
              <div className="stat-label">Best Fitness</div>
            </div>
            
            <div className="stat-item">
              <div className="stat-value">
                {systemMetrics.systemLoad ? systemMetrics.systemLoad.toFixed(0) + '%' : '0%'}
              </div>
              <div className="stat-label">System Load</div>
            </div>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      {!collapsed && (
        <div className="sidebar-actions">
          <button className="action-btn primary">
            <i className="fas fa-plus"></i>
            New Session
          </button>
          
          <button className="action-btn secondary">
            <i className="fas fa-download"></i>
            Export Data
          </button>
        </div>
      )}

      {/* Footer */}
      <div className="sidebar-footer">
        {!collapsed && (
          <div className="footer-content">
            <div className="version-info">
              <div className="version">v1.0.0</div>
              <div className="build">Build 2024.01</div>
            </div>
            
            <div className="footer-links">
              <a href="#" title="Documentation">
                <i className="fas fa-book"></i>
              </a>
              <a href="#" title="Settings">
                <i className="fas fa-cog"></i>
              </a>
              <a href="#" title="Help">
                <i className="fas fa-question-circle"></i>
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Sidebar;