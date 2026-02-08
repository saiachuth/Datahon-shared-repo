---
description: Deploy Streamlit app to Streamlit Cloud
---

# Deploying Your Seattle Accessibility App to Streamlit Cloud

## Option 1: Streamlit Community Cloud (Recommended - FREE)

This is the easiest and most popular way to deploy Streamlit apps for free!

### Prerequisites
1. Your code must be in a **public GitHub repository**
2. You need a GitHub account
3. You need a Streamlit Community Cloud account (free, sign up with GitHub)

### Steps to Deploy

#### 1. Ensure your code is pushed to GitHub
```bash
# Check current status
git status

# Add any changes
git add .
git commit -m "Prepare for deployment"
git push origin main
```

#### 2. Create a Python version file (recommended)
Create a file called `.python-version` in your project root:
```
3.10
```
This tells Streamlit Cloud which Python version to use.

#### 3. Update requirements.txt
Make sure all dependencies are listed. Your current `requirements.txt` should include:
```
streamlit
pandas
folium
streamlit-folium
mapclassify
osmnx
streamlit-searchbox
matplotlib
networkx
shapely
plotly
pydeck
requests
```

**NOTE**: You may need to add specific versions if you encounter issues, e.g.:
```
streamlit>=1.28.0
pandas>=2.0.0
```

#### 4. Check file sizes
⚠️ **IMPORTANT**: Streamlit Community Cloud has limitations:
- Files must be < 100MB each
- Total repo size should be < 1GB

Your files that might be too large:
- `seattle_graph_*.pkl` files (34MB each - should be OK)
- `data_vis_gnn_map.ipynb` (42MB - might cause issues)
- `accessibility_map.html` (27MB)
- `test_map.html` (19MB)
- CSV files (~9MB each)

**Recommendation**: Add large files to `.gitignore` if they're not needed:
```
# Add to .gitignore
*.html
data_vis_gnn_map.ipynb
*.ipynb
```

OR use Git LFS (Large File Storage) for large files.

#### 5. Deploy to Streamlit Community Cloud

1. Go to **https://share.streamlit.io/**
2. Click **"New app"**
3. Connect your GitHub account (if not already connected)
4. Select your repository: `saiachuth/Datahon-shared-repo`
5. Select branch: `main` (or your default branch)
6. Set main file path: `main.py`
7. Click **"Deploy!"**

That's it! Streamlit will:
- Install dependencies from `requirements.txt`
- Run your app
- Provide you with a public URL like: `https://your-app-name.streamlit.app/`

#### 6. Configure Secrets (if needed)
If you have API keys or secrets:
1. Go to your app settings on Streamlit Cloud
2. Click "Secrets" in the sidebar
3. Add your secrets in TOML format:
```toml
LOCATIONIQ_API_KEY = "your-api-key-here"
```

Then update your code to use:
```python
import streamlit as st
api_key = st.secrets["LOCATIONIQ_API_KEY"]
```

---

## Option 2: Render (Alternative FREE option)

### Steps:
1. Create a `render.yaml` file in your project root:
```yaml
services:
  - type: web
    name: seattle-accessibility-app
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
```

2. Go to **https://render.com/**
3. Sign up/Login with GitHub
4. Click "New +" → "Web Service"
5. Connect your GitHub repository
6. Render will auto-detect the `render.yaml` and configure everything
7. Click "Create Web Service"

---

## Option 3: Heroku (Paid after free tier expires)

### Steps:
1. Install Heroku CLI:
```bash
brew install heroku/brew/heroku
```

2. Login to Heroku:
```bash
heroku login
```

3. Create necessary files:

**Procfile** (no extension):
```
web: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0
```

**setup.sh**:
```bash
mkdir -p ~/.streamlit/
echo "[server]
port = $PORT
enableCORS = false
headless = true
[browser]
gatherUsageStats = false
" > ~/.streamlit/config.toml
```

4. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

---

## Troubleshooting Common Issues

### Issue 1: App crashes due to memory
**Solution**: Reduce the number of edges plotted on the map, or use data caching more aggressively.

### Issue 2: Missing dependencies
**Solution**: Run locally first with:
```bash
pip freeze > requirements.txt
```
This captures ALL dependencies with exact versions.

### Issue 3: Files too large
**Solution**: 
- Use `.gitignore` to exclude large files
- Generate large files (like graphs) on the server instead of committing them
- Use Git LFS for files 50MB-100MB

### Issue 4: API keys not working
**Solution**: Use Streamlit secrets management (see Option 1, Step 6)

---

## Best Practices

1. **Test locally first**: Always run `streamlit run main.py` locally before deploying
2. **Use caching**: You're already using `@st.cache_resource` which is great!
3. **Optimize data**: Reduce file sizes, use parquet instead of CSV for large datasets
4. **Monitor performance**: Check Streamlit Cloud logs for errors
5. **Version control**: Keep your GitHub repo clean and well-organized

---

## Next Steps After Deployment

1. **Share your app**: Get the public URL and share it!
2. **Monitor usage**: Check Streamlit Cloud analytics
3. **Update app**: Just push to GitHub - it auto-deploys!
4. **Custom domain**: You can add a custom domain in Streamlit Cloud settings
