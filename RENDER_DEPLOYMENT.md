# 🚀 Deploying to Render.com

This guide explains how to deploy the Thermal Anomaly Detection API to Render.com using Docker.

## 📋 Prerequisites

1. **GitHub Account** - Your code must be in a GitHub repository
2. **Render Account** - Sign up at [render.com](https://render.com)
3. **Docker Files** - Already included:
   - `Dockerfile` - Docker image configuration
   - `.dockerignore` - Files to exclude from image
   - `render.yaml` - Render deployment configuration

## 🐳 Docker Setup

### Files Included

#### 1. `Dockerfile`
- Uses Python 3.11 slim base image
- Installs system dependencies for OpenCV
- Copies application code and ML model
- Exposes port 10000 (Render's default)
- Runs FastAPI with uvicorn

#### 2. `.dockerignore`
- Excludes unnecessary files (datasets, test files, venv, etc.)
- Reduces Docker image size
- Speeds up build time

#### 3. `render.yaml`
- Infrastructure-as-Code configuration
- Defines web service settings
- Configures auto-deploy from git

## 📦 What Gets Deployed

### Included in Docker Image:
✅ API code (`api.py`)  
✅ Configuration (`config.yaml`)  
✅ ML analysis module (`ML_analysis/`)  
✅ Thermal analysis module (`heat_point_analysis/`)  
✅ Trained ML model (`best_model.pth`)  
✅ Python dependencies (`requirements-api.txt`)

### Excluded (too large):
❌ Dataset images  
❌ Virtual environment  
❌ Test files and results  
❌ Documentation  

## 🚀 Deployment Steps

### Option A: Using Render Dashboard (Recommended for First Deploy)

1. **Push Code to GitHub**
   ```bash
   git add Dockerfile .dockerignore render.yaml api.py
   git commit -m "Add Docker configuration for Render deployment"
   git push origin main
   ```

2. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

3. **Create New Web Service**
   - Click **"New +"** → **"Web Service"**
   - Connect your GitHub repository
   - Select **"Arbit-Transformer-Analysis"** repo

4. **Configure Service**
   ```
   Name: thermal-anomaly-api
   Environment: Docker
   Region: Oregon (or your preferred region)
   Branch: main
   Plan: Free (or Starter for paid)
   ```

5. **Advanced Settings**
   - **Docker Build Context**: `./`
   - **Dockerfile Path**: `./Dockerfile`
   - **Health Check Path**: `/health`
   - Leave PORT as auto-assigned

6. **Deploy**
   - Click **"Create Web Service"**
   - Render will:
     - Pull your code
     - Build Docker image
     - Deploy container
     - Assign public URL

7. **Monitor Deployment**
   - Watch build logs in Render dashboard
   - First build takes ~5-10 minutes
   - Subsequent builds are faster (cached layers)

### Option B: Using render.yaml (Infrastructure as Code)

1. **Push render.yaml to GitHub**
   ```bash
   git add render.yaml
   git commit -m "Add Render IaC configuration"
   git push origin main
   ```

2. **Create Service from Blueprint**
   - In Render dashboard, click **"New +"** → **"Blueprint"**
   - Connect repository
   - Render will auto-detect `render.yaml`
   - Click **"Apply"**

## 🔗 Access Your API

Once deployed, Render provides a URL like:
```
https://thermal-anomaly-api.onrender.com
```

### Test Endpoints:

**Health Check:**
```bash
curl https://thermal-anomaly-api.onrender.com/health
```

**Detect Anomalies:**
```bash
curl -X POST \
  -F "baseline=@baseline.jpg" \
  -F "maintenance=@maintenance.jpg" \
  -F "transformer_id=T1_Test" \
  -F "return_format=json" \
  https://thermal-anomaly-api.onrender.com/detect
```

**Interactive Docs:**
```
https://thermal-anomaly-api.onrender.com/docs
```

## ⚙️ Configuration

### Environment Variables

Set in Render Dashboard → Environment tab:

| Variable | Value | Description |
|----------|-------|-------------|
| `PORT` | Auto-assigned | Render sets this automatically |
| `PYTHON_VERSION` | 3.11 | Python version |

### Custom Configuration

To modify detection parameters, update `config.yaml` and redeploy:

```yaml
detection:
  ml:
    threshold: 0.5
    min_area: 200
    max_area: 5000
  thermal:
    temperature_threshold: 200
    min_cluster_size: 15
```

## 💰 Pricing

### Free Tier
- ✅ 750 hours/month (enough for 1 always-on service)
- ✅ 512 MB RAM
- ✅ 0.1 CPU
- ✅ Automatic SSL certificate
- ⚠️ Spins down after 15 minutes of inactivity
- ⚠️ Cold start takes ~30 seconds

### Starter Tier ($7/month)
- ✅ Always on (no spin down)
- ✅ 512 MB RAM
- ✅ 0.5 CPU
- ✅ Faster response times

### Standard Tier ($25/month)
- ✅ 2 GB RAM
- ✅ 1 CPU
- ✅ Better for production

## 🔄 Auto-Deploy

With `render.yaml`, every push to `main` branch triggers auto-deploy:

```bash
# Make changes
vim api.py

# Commit and push
git add api.py
git commit -m "Update API endpoint"
git push origin main

# Render automatically rebuilds and deploys
```

## 📊 Monitoring

### View Logs
```
Render Dashboard → Your Service → Logs
```

### Metrics Available:
- CPU usage
- Memory usage
- Request count
- Response times
- Error rates

## 🐛 Troubleshooting

### Build Fails

**Issue:** Docker build fails
```
Error: Failed to build image
```

**Solution:**
1. Check `Dockerfile` syntax
2. Verify all files exist in repo
3. Check build logs for specific errors

### Service Won't Start

**Issue:** Service crashes on startup
```
Error: Application failed to start
```

**Solution:**
1. Check that `best_model.pth` is committed to repo
2. Verify `requirements-api.txt` has all dependencies
3. Check logs for missing imports

### Out of Memory

**Issue:** Service crashes with OOM
```
Error: Container killed due to memory limit
```

**Solution:**
1. Upgrade to Starter or Standard plan (more RAM)
2. Optimize model loading (load once, not per request)
3. Reduce image size in requests

### Cold Start Delays

**Issue:** First request after inactivity is slow

**Solution:**
1. Free tier spins down after 15 min
2. Upgrade to Starter plan ($7/mo) for always-on
3. Use health check pings to keep alive (not recommended on free tier)

### Port Issues

**Issue:** Service not responding
```
Error: Failed to bind to port
```

**Solution:**
- Ensure API uses `PORT` env variable
- Don't hardcode port 8000
- Use: `port = int(os.environ.get("PORT", 8000))`

## 🔐 Security Best Practices

1. **Don't commit sensitive data**
   - No API keys in code
   - Use Render's secret environment variables

2. **CORS Configuration**
   - Update `api.py` CORS settings for production
   - Restrict allowed origins

3. **Rate Limiting**
   - Consider adding rate limiting middleware
   - Protect against abuse

## 📈 Scaling

### Horizontal Scaling (Multiple Instances)
```yaml
# In render.yaml (Starter plan or higher)
numInstances: 2
```

### Vertical Scaling (More Resources)
```yaml
# In render.yaml
plan: standard  # 2GB RAM, 1 CPU
```

## 🔗 Custom Domain

1. Go to **Settings** → **Custom Domain**
2. Add your domain (e.g., `api.yourdomain.com`)
3. Update DNS records as instructed
4. Render provisions SSL automatically

## 📝 Useful Commands

### Build Docker Image Locally (Testing)
```bash
docker build -t thermal-api .
docker run -p 8000:8000 thermal-api
```

### Check Image Size
```bash
docker images thermal-api
# Optimize if > 1GB
```

### Test Docker Container
```bash
docker run -p 8000:8000 thermal-api
curl http://localhost:8000/health
```

## 🎯 Production Checklist

- [ ] Code pushed to GitHub `main` branch
- [ ] `Dockerfile`, `.dockerignore`, `render.yaml` included
- [ ] ML model (`best_model.pth`) committed to repo
- [ ] `requirements-api.txt` up to date
- [ ] CORS settings configured for production
- [ ] Health check endpoint working (`/health`)
- [ ] Test API locally with Docker
- [ ] Deploy to Render
- [ ] Test deployed API
- [ ] Monitor logs for errors
- [ ] Set up custom domain (optional)
- [ ] Configure auto-deploy from git

## 📚 Additional Resources

- [Render Documentation](https://render.com/docs)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Render Free Tier Limits](https://render.com/docs/free)

## 🆘 Support

- **Render Support**: [render.com/support](https://render.com/support)
- **Community Forum**: [community.render.com](https://community.render.com)
- **Status Page**: [status.render.com](https://status.render.com)

---

## ✅ Summary

Your API is now ready to deploy to Render.com! The Docker setup ensures:
- ✨ Consistent environment across dev and production
- 🚀 Fast deployment with cached layers
- 📦 Minimal image size with `.dockerignore`
- 🔄 Auto-deploy on git push
- 💰 Free tier available for testing

**Deployment Time:** ~5-10 minutes for first deploy  
**Your API URL:** `https://your-service-name.onrender.com`
