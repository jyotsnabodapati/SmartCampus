## 🎓 SmartCampus Pull Request

### 📌 Summary of Changes
<!-- Provide a brief description of what this PR introduces, fixes, or improves. -->

### 🏷️ PR Type
- [ ] 🚀 New Feature (RAG pipeline enhancement, new knowledge documents)
- [ ] 🐛 Bugfix (fixing an error, regression, or edge case)
- [ ] 🧪 Testing (adding or improving Pytest tests or benchmarks)
- [ ] 🐳 DevOps & CI/CD (Docker, GitHub Actions, deployment configurations)
- [ ] 📝 Documentation (updating README, diagrams, or viva questions)
- [ ] 🔒 Security & Code Quality (secret protection, dependency audit)

---

### 🧪 Pre-Merge Verification Checklist
- [ ] **Pytest Unit & Integration Tests:** Ran `pytest -v` and all tests passed.
- [ ] **15-Question Benchmark:** Ran `python test_rag.py` and scored 100% accuracy.
- [ ] **Smoke Test Verified:** Ran `python smoke_test.py` and all checks passed.
- [ ] **No Secrets Committed:** Verified `.env`, Groq API keys, and local DB files are not staged.
- [ ] **Streamlit UI Functional:** Tested locally with `streamlit run app.py`.
- [ ] **Docker Build:** Verified `Dockerfile` and `docker-compose.yml` build cleanly.

---

### 📷 Evidence & Test Output
<!-- Paste command output or screenshot evidence of passing tests. -->
```text
============================= 40 passed in 40.50s =============================
```

### 👥 Code Reviewers
- Assigned to `@jyotsnabodapati` for final approval.

