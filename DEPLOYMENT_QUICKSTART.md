# 🚀 Quick Deployment Guide - Render.com

## 📦 Files Created

```
✅ Dockerfile          - Docker image configuration
✅ .dockerignore       - Exclude unnecessary files  
✅ render.yaml         - Render IaC configuration
✅ test_docker.sh      - Local testing script
✅ RENDER_DEPLOYMENT.md - Complete deployment guide
```

## ⚡ Quick Start (3 Steps)

### 1️⃣ Test Locally
```bash
# Build and test Docker container
./test_docker.sh

# Or manually:
docker build -t thermal-api .
docker run -p 8000:8000 -e PORT=8000 thermal-api

# Test API
curl http://localhost:8000/health
```

### 2️⃣ Push to GitHub
```bash
git add Dockerfile .dockerignore render.yaml api.py
git commit -m "Add Docker configuration for Render"
git push origin main
```

### 3️⃣ Deploy to Render
1. Go to [render.com](https://render.com)
2. New → Web Service
3. Connect GitHub repo
4. Select **Docker** environment
5. Click **Create Web Service**

**Done! ✨** Your API will be live at: `https://your-app.onrender.com`

---

## 🔧 Configuration Changes Made

### api.py
```python
# Now reads PORT from environment variable
port = int(os.environ.get("PORT", 8000))
```

### Dockerfile
- Base: Python 3.11 slim
- Includes: OpenCV system dependencies
- Copies: API code, ML models, configs
- Port: 10000 (Render default)
- CMD: `uvicorn api:app --host 0.0.0.0 --port ${PORT}`

### render.yaml
- Service type: Web Service (Docker)
- Plan: Free tier (750h/month)
- Auto-deploy: On git push
- Health check: `/health`

---

## 📊 What Gets Deployed

| Included ✅ | Excluded ❌ |
|------------|------------|
| `api.py` | `Dataset/` (images) |
| `config.yaml` | `thermal_env/` |
| `ML_analysis/` | `test_*.py` |
| `heat_point_analysis/` | Results folders |
| `best_model.pth` | Documentation |
| `requirements-api.txt` | `*.md` files |

---

## 🌐 Access Your Deployed API

Once deployed on Render:

**Base URL:**
```
https://thermal-anomaly-api.onrender.com
```

**Endpoints:**
```bash
# Health check
curl https://your-app.onrender.com/health

# Interactive docs
https://your-app.onrender.com/docs

# Detect anomalies
curl -X POST \
  -F "baseline=@baseline.jpg" \
  -F "maintenance=@maintenance.jpg" \
  -F "transformer_id=T1" \
  -F "return_format=json" \
  https://your-app.onrender.com/detect
```

---

## 💰 Render Free Tier

| Feature | Free Tier |
|---------|-----------|
| Hours/month | 750 (enough for 1 service) |
| RAM | 512 MB |
| CPU | 0.1 |
| Bandwidth | 100 GB/month |
| SSL | ✅ Free HTTPS |
| Auto-deploy | ✅ On git push |
| Spin down | After 15 min inactivity |
| Cold start | ~30 seconds |

**Upgrade to Starter ($7/mo) for:**
- Always-on (no spin down)
- Faster response times
- 0.5 CPU instead of 0.1

---

## 🐛 Common Issues

### Issue: Build fails
**Solution:** Check that `best_model.pth` is committed to git

### Issue: Out of memory
**Solution:** Upgrade to Starter plan (more RAM)

### Issue: Slow first request
**Solution:** Free tier spins down after 15 min. Upgrade for always-on.

### Issue: Port binding error
**Solution:** Render auto-assigns PORT. API now reads from env variable ✅

---

## 🔄 Update & Redeploy

```bash
# Make changes
vim api.py

# Commit and push
git add .
git commit -m "Update API"
git push origin main

# Render auto-deploys! 🚀
```

Monitor deployment in Render dashboard → Logs

---

## 📋 Deployment Checklist

- [ ] Test Docker build locally (`./test_docker.sh`)
- [ ] Verify `/health` endpoint works
- [ ] Commit all files to git
- [ ] Push to GitHub `main` branch
- [ ] Create Render web service
- [ ] Select Docker environment
- [ ] Wait for build (~5-10 min first time)
- [ ] Test deployed API
- [ ] Check logs for errors
- [ ] Update CORS settings for production

---

## 🎯 Production Tips

1. **Security:** Update CORS in `api.py` for production domains
2. **Monitoring:** Use Render dashboard for logs and metrics
3. **Scaling:** Upgrade plan if traffic increases
4. **Custom Domain:** Add in Render settings (free SSL included)
5. **Environment Variables:** Set secrets in Render dashboard, not code

---

## 📚 Documentation

- 📖 Complete guide: `RENDER_DEPLOYMENT.md`
- 🐳 Docker config: `Dockerfile`
- 🏗️ IaC config: `render.yaml`
- 🧪 Test script: `test_docker.sh`

---

## ✅ Status

**Current State:**
- ✅ Docker configuration complete
- ✅ Render deployment files ready
- ✅ API updated for cloud deployment
- ✅ Test script available
- ✅ Documentation created

**Next Action:**
```bash
# Test locally
./test_docker.sh

# Push to GitHub
git push origin main

# Deploy on Render
# → render.com → New Web Service
```

**Estimated Deploy Time:** 5-10 minutes

---

## 🆘 Need Help?

- Render Docs: https://render.com/docs
- Community: https://community.render.com
- Status: https://status.render.com

---

**Your API is ready to deploy! 🚀**
