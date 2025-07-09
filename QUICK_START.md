# 🚀 Quick Start Guide - Darwin-Gödel Machine

## Instalasi Cepat (5 Menit)

### 1. Clone & Setup
```bash
git clone https://github.com/nsjjks77/jahus.git
cd jahus/web_interface
npm install
```

### 2. Start Server
```bash
npm start
```

### 3. Akses Interface
Buka browser: `http://localhost:12000`

## Demo Cepat

### Membuat Evolution Session Pertama

1. **Klik "Evolution Control"** di sidebar kiri
2. **Klik "+ New Session"**
3. **Gunakan setting default** atau sesuaikan:
   - Population Size: `100`
   - Mutation Rate: `0.1`
   - Selection Strategy: `NSGA-II`
   - Max Generations: `50`
4. **Klik "Start Evolution"**
5. **Lihat real-time progress** di dashboard

### Monitoring Real-time

1. **Dashboard**: Lihat system stats dan metrics
2. **Switch Metrics**: Klik `FITNESS/DIVERSITY/PERFORMANCE`
3. **System Load**: Monitor resource usage (warna berubah sesuai load)
4. **Multiple Sessions**: Jalankan beberapa session bersamaan

### Tips Cepat

- **Green System Load** (0-50%): Optimal performance
- **Yellow System Load** (50-80%): Good performance
- **Red System Load** (80-100%): High load, consider reducing population

### Troubleshooting Cepat

**Server tidak start?**
```bash
sudo kill -9 $(lsof -t -i:12000)
npm start
```

**Performance lambat?**
- Reduce population size ke 50-100
- Close unused browser tabs
- Restart server

---
**Ready to evolve? Let's go! 🧬**