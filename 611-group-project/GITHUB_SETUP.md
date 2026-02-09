# Create a New GitHub Repository for This Project

Follow these steps to put this project in a **new** GitHub repository.

## 1. Create the repository on GitHub

1. Go to [github.com](https://github.com) and sign in.
2. Click the **+** (top right) → **New repository**.
3. Fill in:
   - **Repository name:** `611-group-project` (or e.g. `mgta611-deep-learning`)
   - **Description:** `MGTA 611 Group Project — Deep Learning for Retail Sales Forecasting`
   - **Visibility:** Public or Private (your choice).
   - **Do not** check “Add a README”, “Add .gitignore”, or “Choose a license” (this folder already has them).
4. Click **Create repository**.

## 2. Push this folder to the new repo

Open a terminal in **this folder** (the one containing `611 Script.ipynb`, `Data/`, `README.md`).

### If this folder is not yet a git repo

```bash
cd "C:\Users\tabos\.cursor\worktrees\NS-3_Sumo\umi\611-group-project"

git init
git add .
git commit -m "Initial commit: MGTA 611 Group Project - Deep Learning for Retail Sales"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/611-group-project.git
git push -u origin main
```

Replace **YOUR_USERNAME** with your GitHub username (and `611-group-project` with your repo name if you chose a different one).

### If you use SSH

```bash
git remote add origin git@github.com:YOUR_USERNAME/611-group-project.git
git push -u origin main
```

### If this folder is already part of another git repo

If you’re inside the `NS-3_Sumo` worktree, run the commands above from a **copy** of this project (e.g. a new folder that contains only the 611 project files), then `git init` there and push.  
Alternatively, create the new repo and add it as a separate remote, then push only this project’s files (e.g. from a clean copy).

## 3. After pushing

- Your project will be at: `https://github.com/YOUR_USERNAME/611-group-project`
- Share this URL with your team or add them as collaborators (Settings → Collaborators).
- For large `Data/*.csv` files: if the push fails due to size, see [GitHub’s file size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files). You can use [Git LFS](https://git-lfs.github.com/) or add `Data/*.csv` to `.gitignore` and document how to obtain the data in the README.

You’re done. The new repository is created and this project is pushed to it.
