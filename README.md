# Darwin-Gödel Machine (DGM) - Advanced Interface

![DGM Logo](https://img.shields.io/badge/DGM-Advanced%20Interface-blue?style=for-the-badge&logo=react)
![Version](https://img.shields.io/badge/version-1.0.0-green?style=for-the-badge)
![Build](https://img.shields.io/badge/build-2024.01-orange?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-red?style=for-the-badge)

## 🧠 Tentang Darwin-Gödel Machine

Darwin-Gödel Machine (DGM) adalah sistem AI self-improving yang menggabungkan prinsip evolusi Darwin dengan kemampuan self-modification Gödel. Sistem ini mampu mengoptimalkan dirinya sendiri melalui proses evolusi yang terkontrol, menciptakan agen-agen AI yang semakin cerdas dari generasi ke generasi.

### 🎯 Konsep Utama

- **Darwinian Evolution**: Seleksi alam untuk mengoptimalkan performa agen
- **Gödelian Self-Modification**: Kemampuan sistem untuk memodifikasi kode dirinya sendiri
- **Multi-Objective Optimization**: Optimasi berdasarkan multiple criteria (fitness, diversity, performance)
- **Real-time Evolution**: Monitoring dan kontrol evolusi secara real-time

## ✨ Fitur Utama

### 🎛️ Dashboard
- **Real-time System Monitoring**: CPU load, memory usage, active sessions
- **Live Metrics Visualization**: Interactive charts untuk fitness, diversity, dan performance
- **Session Management**: Overview semua evolution sessions yang aktif
- **System Health Indicators**: Status indicators dengan color coding

### 🧬 Evolution Control
- **Advanced Parameter Configuration**: 
  - Population size (50-500)
  - Mutation rate (0.01-0.5)
  - Selection strategies (Tournament, Roulette, NSGA-II)
  - Crossover methods (Single-point, Multi-point, Uniform)
- **Real-time Evolution Tracking**: Live updates selama proses evolusi
- **Multi-session Support**: Jalankan multiple evolution sessions bersamaan
- **Generation History**: Track progress setiap generasi

### 👥 Agent Viewer
- **Agent Browser**: Explore dan analyze individual agents
- **Performance Analytics**: Detailed metrics untuk setiap agent
- **Code Inspection**: View generated code dari agents
- **Genealogy Tracking**: Track lineage dan evolution history

### 💻 Code Editor
- **Syntax Highlighting**: Advanced code editor dengan syntax highlighting
- **Auto-completion**: Intelligent code completion
- **Real-time Validation**: Instant code validation dan error detection
- **Version Control**: Track changes dan revisions

### 📊 Analytics
- **Performance Metrics**: Comprehensive analytics dashboard
- **Trend Analysis**: Historical performance trends
- **Comparative Analysis**: Compare different evolution runs
- **Export Capabilities**: Export data untuk further analysis

### 🤖 LLM Interface
- **AI-Powered Assistance**: Integration dengan Large Language Models
- **Natural Language Queries**: Query system menggunakan natural language
- **Code Generation**: AI-assisted code generation
- **Intelligent Suggestions**: Smart recommendations untuk optimization

### 🖥️ System Monitor
- **Resource Monitoring**: Real-time system resource usage
- **Performance Profiling**: Detailed performance profiling
- **Alert System**: Automated alerts untuk system issues
- **Health Diagnostics**: Comprehensive system health checks

## 🛠️ Teknologi yang Digunakan

### Backend
- **Node.js**: Runtime environment
- **Express.js**: Web framework
- **Socket.io**: Real-time WebSocket communication
- **OpenAI API**: LLM integration

### Frontend
- **React**: UI framework
- **Modern JavaScript (ES6+)**: Latest JavaScript features
- **CSS3**: Advanced styling dengan animations
- **Chart.js**: Data visualization
- **Font Awesome**: Icon library

### Build Tools
- **Webpack**: Module bundler
- **Babel**: JavaScript transpiler
- **npm**: Package manager

## 🚀 Instalasi dan Setup

### Prerequisites
- Node.js (v14 atau lebih baru)
- npm atau yarn
- OpenAI API key (opsional, untuk LLM features)

### 1. Clone Repository
```bash
git clone https://github.com/nsjjks77/jahus.git
cd jahus
```

### 2. Install Dependencies
```bash
# Install backend dependencies
cd web_interface
npm install
```

### 3. Environment Setup
```bash
# Set OpenAI API key (opsional)
export OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Build Frontend
```bash
# Build frontend assets
npm run build
```

### 5. Start Server
```bash
# Start the DGM web interface
npm start
```

Server akan berjalan di: `http://localhost:12000`

## 📖 Cara Penggunaan

### 1. Akses Web Interface
Buka browser dan navigasi ke `http://localhost:12000`

### 2. Dashboard Overview
- **System Stats**: Lihat overview sistem (agents, sessions, fitness, load)
- **Real-time Metrics**: Monitor performa sistem secara real-time
- **Quick Actions**: Akses cepat untuk create session atau export data

### 3. Membuat Evolution Session Baru

#### Langkah-langkah:
1. **Klik "Evolution Control"** di sidebar
2. **Klik "+ New Session"** 
3. **Konfigurasi Parameters**:
   ```
   Population Size: 100-200 (recommended)
   Mutation Rate: 0.1-0.3 (recommended)
   Selection Strategy: NSGA-II (untuk multi-objective)
   Crossover Method: Multi-point (recommended)
   Max Generations: 50-100
   ```
4. **Klik "Start Evolution"**
5. **Monitor Progress**: Lihat real-time updates di dashboard

### 4. Monitoring Evolution

#### Real-time Tracking:
- **Generation Counter**: Track current generation
- **Fitness Progress**: Monitor fitness improvement
- **Population Diversity**: Track genetic diversity
- **System Load**: Monitor resource usage

#### Metrics Switching:
- Klik tombol **FITNESS/DIVERSITY/PERFORMANCE** untuk switch metrics
- Charts akan update secara real-time

### 5. Session Management

#### Multiple Sessions:
- Jalankan multiple evolution sessions bersamaan
- Setiap session memiliki ID unik
- Monitor semua sessions dari dashboard

#### Session Controls:
- **Pause/Resume**: Control evolution process
- **Stop**: Terminate session
- **Export**: Export session data

### 6. Advanced Features

#### Agent Analysis:
1. Klik **"Agent Viewer"**
2. Browse individual agents
3. Analyze performance metrics
4. View generated code

#### Code Editing:
1. Klik **"Code Editor"**
2. Edit agent code directly
3. Real-time validation
4. Save changes

#### Analytics:
1. Klik **"Analytics"**
2. View detailed performance analytics
3. Compare different runs
4. Export data for analysis

## 🏗️ Struktur Project

```
jahus/
├── web_interface/
│   ├── server.js              # Main server file
│   ├── public/
│   │   ├── index.html         # Main HTML template
│   │   └── bundle.js          # Compiled frontend
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   ├── components/        # React components
│   │   └── styles/            # CSS styles
│   ├── package.json           # Dependencies
│   └── webpack.config.js      # Webpack configuration
├── core/                      # Core DGM algorithms
├── agents/                    # Agent implementations
├── evolution/                 # Evolution algorithms
├── utils/                     # Utility functions
├── examples/                  # Usage examples
├── package.json              # Main package.json
└── README.md                 # This file
```

## 🔌 API Endpoints

### Evolution Management
```javascript
// Start new evolution session
POST /api/evolution/start
Body: {
  populationSize: 100,
  mutationRate: 0.1,
  selectionStrategy: "nsga2",
  crossoverMethod: "multipoint",
  maxGenerations: 50
}

// Get session status
GET /api/evolution/session/:id

// Stop evolution session
POST /api/evolution/stop/:id

// Get all sessions
GET /api/evolution/sessions
```

### Metrics and Monitoring
```javascript
// Get real-time metrics
GET /api/metrics/realtime

// Get system stats
GET /api/system/stats

// Get evolution history
GET /api/evolution/history/:sessionId
```

### Agent Management
```javascript
// Get all agents
GET /api/agents

// Get specific agent
GET /api/agents/:id

// Update agent code
PUT /api/agents/:id
Body: { code: "agent code here" }
```

## 🔧 WebSocket Events

### Client → Server
```javascript
// Join evolution session
socket.emit('join-session', { sessionId: 'abc123' });

// Request metrics update
socket.emit('request-metrics', { type: 'fitness' });

// Control evolution
socket.emit('evolution-control', { 
  action: 'pause', 
  sessionId: 'abc123' 
});
```

### Server → Client
```javascript
// Evolution updates
socket.on('evolution-update', (data) => {
  // Handle evolution progress
});

// Metrics updates
socket.on('metrics-update', (data) => {
  // Handle real-time metrics
});

// System status
socket.on('system-status', (data) => {
  // Handle system status changes
});
```

## 🎨 Customization

### Themes
Edit `/web_interface/src/styles/main.css` untuk customize appearance:
```css
:root {
  --primary-color: #3498db;
  --secondary-color: #2c3e50;
  --success-color: #27ae60;
  --warning-color: #f39c12;
  --danger-color: #e74c3c;
}
```

### Evolution Parameters
Modify default parameters di `/web_interface/server.js`:
```javascript
const DEFAULT_CONFIG = {
  populationSize: 100,
  mutationRate: 0.1,
  selectionStrategy: 'tournament',
  crossoverMethod: 'singlepoint',
  maxGenerations: 50
};
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Server tidak start
```bash
# Check port availability
netstat -tulpn | grep :12000

# Kill process if needed
sudo kill -9 $(lsof -t -i:12000)

# Restart server
cd web_interface && npm start
```

#### 2. WebSocket connection failed
```bash
# Check firewall settings
sudo ufw status

# Allow port 12000
sudo ufw allow 12000
```

#### 3. High memory usage
```bash
# Monitor memory usage
htop

# Restart server to clear memory
cd web_interface && npm restart
```

#### 4. Evolution session stuck
- Check system load di dashboard
- Reduce population size jika terlalu tinggi
- Restart session jika diperlukan

### Performance Optimization

#### 1. Reduce Population Size
Untuk sistem dengan resource terbatas:
```javascript
populationSize: 50-100  // Instead of 200-500
```

#### 2. Adjust Update Frequency
Edit WebSocket update interval:
```javascript
const UPDATE_INTERVAL = 2000; // 2 seconds instead of 1
```

## 📊 Monitoring dan Logging

### System Logs
```bash
# View server logs
tail -f logs/server.log

# View evolution logs
tail -f logs/evolution.log

# View error logs
tail -f logs/error.log
```

### Performance Metrics
- **CPU Usage**: Monitor via dashboard
- **Memory Usage**: Real-time monitoring
- **Evolution Speed**: Generations per minute
- **Session Count**: Active sessions

## 🤝 Contributing

### Development Setup
```bash
# Fork repository
git fork https://github.com/nsjjks77/jahus.git

# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git commit -am "Add new feature"

# Push to branch
git push origin feature/new-feature

# Create Pull Request
```

### Code Style
- Use ES6+ features
- Follow React best practices
- Add comments untuk complex logic
- Write tests untuk new features

### Testing
```bash
# Run tests
npm test

# Run specific test
npm test -- --grep "evolution"

# Coverage report
npm run coverage
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Darwin's Theory of Evolution**: Inspiration untuk selection algorithms
- **Gödel's Incompleteness Theorems**: Foundation untuk self-modification
- **OpenAI**: LLM integration support
- **React Community**: UI framework dan components
- **Node.js Community**: Backend runtime dan packages

## 📞 Support

### Documentation
- [Wiki](https://github.com/nsjjks77/jahus/wiki)
- [API Documentation](https://github.com/nsjjks77/jahus/docs/api)
- [Examples](https://github.com/nsjjks77/jahus/examples)

### Community
- [Discord Server](https://discord.gg/dgm-community)
- [GitHub Discussions](https://github.com/nsjjks77/jahus/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/darwin-godel-machine)

### Contact
- **Email**: support@dgm-project.org
- **Twitter**: [@DGMProject](https://twitter.com/DGMProject)
- **LinkedIn**: [DGM Project](https://linkedin.com/company/dgm-project)

---

**Made with ❤️ by the DGM Team**

*"Evolution never stops, and neither do we."*

## Referensi

Proyek ini terinspirasi oleh konsep Darwin-Gödel Machine yang dijelaskan dalam blog:
https://richardcsuwandi.github.io/blog/2025/dgm/