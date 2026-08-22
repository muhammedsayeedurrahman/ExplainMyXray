# Team Git Workflow - Hackathon

## 🌳 Branch Structure

```
main (protected - judges will see this)
  ↓
decisioos (team development - merge here)
  ↓
  ├── feature/frontend-work (Friend 1)
  └── feature/backend-work (Friend 2)
```

## 👥 Team Member Assignments

### Friend 1 - Frontend Developer
**Branch:** `feature/frontend-work`

**Tasks:**
- React components and pages
- UI/UX improvements
- Tailwind CSS styling
- Frontend routing
- Form validations
- Charts and data visualization

**Commands:**
```bash
git checkout feature/frontend-work
git pull origin feature/frontend-work
# ... do your work ...
git add .
git commit -m "feat: your feature description"
git push origin feature/frontend-work
```

**Push Schedule:** Every 1-2 hours max (as per team agreement)

---

### Friend 2 - Backend Developer
**Branch:** `feature/backend-work`

**Tasks:**
- Express API endpoints
- Database models and migrations
- Authentication & authorization
- Business logic
- API testing
- Database seeding

**Commands:**
```bash
git checkout feature/backend-work
git pull origin feature/backend-work
# ... do your work ...
git add .
git commit -m "feat: your feature description"
git push origin feature/backend-work
```

**Push Schedule:** Every 1-2 hours max (as per team agreement)

---

### You - Full Stack + Integration
**Branch:** `decisioos`

**Tasks:**
- Integrate frontend + backend
- Full-stack features
- Testing and bug fixes
- Documentation
- Deployment preparation

**Commands:**
```bash
git checkout decisioos
git pull origin decisioos
# ... do your work ...
git add .
git commit -m "feat: your feature description"
git push origin decisioos
```

---

## 🔄 Workflow

### Daily Workflow

1. **Start of day:**
   ```bash
   git checkout your-branch
   git pull origin your-branch
   git pull origin decisioos  # Get latest changes
   ```

2. **During work:**
   - Make small, frequent commits
   - Push every 1-2 hours
   - Use meaningful commit messages

3. **End of day or feature complete:**
   - Push your final changes
   - Create a Pull Request to `decisioos`
   - Tag team lead for review

### Merging to Main

**ONLY the team lead** merges `decisioos` → `main` when:
- Feature is complete and tested
- Ready to show judges
- All team members have pushed their work

---

## 📝 Commit Message Format

Use conventional commits:

```bash
feat: add user authentication
fix: resolve login bug
refactor: improve code structure
docs: update API documentation
test: add unit tests for auth
style: format code with prettier
```

---

## ⚠️ Important Rules

1. **NEVER** work directly on `main` branch
2. **NEVER** push to `main` without review
3. **ALWAYS** pull before you push
4. **ALWAYS** commit every 1-2 hours max
5. **NEVER** commit sensitive data (.env files, API keys)
6. **ALWAYS** test your code before pushing

---

## 🆘 Common Commands

### Update your branch with latest team changes
```bash
git checkout your-branch
git pull origin decisioos
# Resolve any conflicts
git push origin your-branch
```

### Undo last commit (not pushed yet)
```bash
git reset --soft HEAD~1
```

### See what changed
```bash
git status
git diff
```

### Discard local changes
```bash
git checkout -- filename
```

---

## 📞 Quick Contact

If you face merge conflicts or git issues:
1. Don't panic
2. Take a screenshot
3. Contact team lead
4. Don't force push without approval

---

## 🎯 Hackathon Timeline

**Hourly Commits:** Max 1-2 pushes per hour
**Daily Merges:** End of day merge to `decisioos`
**Final Submission:** Merge `decisioos` → `main` before deadline

---

**Remember:** Clean git history = Professional impression on judges! 🏆
