# Push to GitHub

`gh` (GitHub CLI) is not installed on this machine. Use one of the options below.

## Option A — GitHub website (recommended)

1. Open [https://github.com/new](https://github.com/new)
2. Repository name: `ai-gym-form-coach` (or your choice)
3. **Do not** initialize with README (this repo already has one)
4. Create the repository

Then in PowerShell:

```powershell
cd "c:\Users\USER\OneDrive - University of Haifa\Documents\AI-Coach"

# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/ai-gym-form-coach.git
git branch -M main
git push -u origin main
```

## Option B — Install GitHub CLI later

```powershell
winget install GitHub.cli
gh auth login
gh repo create ai-gym-form-coach --public --source=. --remote=origin --push
```

## Before first push

Ensure all commits are made:

```powershell
git status
git log --oneline -5
```

Expected commits:

- `chore: initialize AI Gym Form Coach project structure`
- `feat(data): add video loading and frame extraction pipeline`
- `feat(pose): add scalable MediaPipe pose extraction pipeline`
