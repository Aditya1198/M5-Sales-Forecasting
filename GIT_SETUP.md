# Git Setup Guide for M5 Forecasting Project

## Issue Identified

Git is not installed or not available in your system PATH.

**Error**: `The term 'git' is not recognized...`

---

## Solution: Install Git

### Option 1: Download Git for Windows (Recommended)

1. **Download** Git from: https://git-scm.com/download/win
2. **Run the installer** (git-2.xx.x-64-bit.exe)
3. **Installation Settings**:
   - ✅ Use default settings (recommended)
   - ✅ Make sure "Git from the command line and also from 3rd-party software" is selected
   - ✅ Use the default branch name "main"
4. **Restart** your terminal/PowerShell after installation
5. **Verify** installation:
   ```bash
   git --version
   ```

### Option 2: Using Winget (Windows Package Manager)

```powershell
winget install Git.Git
```

Then restart your terminal.

### Option 3: Using Chocolatey

```powershell
choco install git
```

Then restart your terminal.

---

## After Installing Git

Once Git is installed, follow these steps to set up your repository:

### Step 1: Configure Git (First Time Only)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 2: Initialize Repository

```bash
cd d:\m5-forecasting-accuracy
git init
```

### Step 3: Create .gitignore

Create a `.gitignore` file to exclude large data files and model files:

```bash
# Create .gitignore
echo "# Large data files
*.csv
!sample_submission.csv

# Model files
*.pkl

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Virtual Environment
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/
" > .gitignore
```

### Step 4: Add Files to Git

```bash
# Add all Python scripts and documentation
git add *.py
git add *.md
git add app/*.py
git add *.bat

# Add gitignore
git add .gitignore
```

### Step 5: Make First Commit

```bash
git commit -m "Initial commit: M5 Forecasting XGBoost model with FastAPI and Streamlit deployment"
```

### Step 6: Connect to GitHub (Optional)

If you want to push to GitHub:

```bash
# Create a new repository on GitHub first, then:
git remote add origin https://github.com/yourusername/m5-forecasting.git
git branch -M main
git push -u origin main
```

---

## Quick Commands Reference

### Check Status
```bash
git status
```

### Add Changes
```bash
git add .                    # Add all files
git add filename.py          # Add specific file
```

### Commit Changes
```bash
git commit -m "Your message"
```

### View History
```bash
git log --oneline
```

### Create Branch
```bash
git checkout -b feature-name
```

### Push to Remote
```bash
git push origin main
```

---

## Files to Commit

Here's what should be in your repository:

### ✅ Should Commit
- `*.py` (Python scripts)
- `*.md` (Documentation)
- `*.bat` (Startup scripts)
- `app/` (Application code)
- `.gitignore`
- `requirements.txt`
- `feature_importance.csv` (small file)

### ❌ Should NOT Commit
- `*.csv` (Large data files - except sample_submission.csv)
- `*.pkl` (Model files - too large)
- `__pycache__/`
- `.venv/` or `venv/`
- `.streamlit/`

---

## Troubleshooting

### Git Still Not Found After Install

**Solution**:
1. Close all terminals/PowerShell windows
2. Open new PowerShell
3. Try: `git --version`
4. If still not working, add Git to PATH manually:
   - Default Git location: `C:\Program Files\Git\cmd`
   - Add to System PATH environment variable

### Permission Denied

**Solution**:
```bash
# Run PowerShell as Administrator
```

### Large Files Error

**Solution**:
```bash
# If you accidentally committed large files
git rm --cached *.csv
git rm --cached *.pkl
git commit -m "Remove large files"
```

---

## Next Steps After Git Setup

1. Install Git following instructions above
2. Restart your terminal
3. Run the initialization commands
4. Start tracking your code changes!

---

**Need help?** Let me know which step you're on!
