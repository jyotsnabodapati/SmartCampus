# 🔄 SmartCampus — Disaster Recovery & Rollback Guide

This document defines standard operating procedures (SOP) for identifying, mitigating, and rolling back failures in the SmartCampus application across Git, Docker, and persistent storage layers.

---

## 1. Rollback Trigger Criteria

Initiate a rollback immediately if any of the following occur in production:
* **Critical Service Outage:** Streamlit health check (`GET /_stcore/health`) returns non-200 or connection refused.
* **RAG Pipeline Failure:** Retrieval fails across all documents or returns unhandled 500 exceptions.
* **Hallucination Regression:** Out-of-domain queries fail to trigger the refusal guardrail (*"I don't have enough information to answer that."*).
* **Storage Corruption:** ChromaDB fails to initialize or collection items cannot be read.

---

## 2. Identifying the Last Working Version

### A. Identify Last Healthy Git Tag
List all tagged releases:
```bash
git tag -l -n
```
View commit details of the previous stable tag:
```bash
git show v0.1.0
```

### B. Identify Last Healthy Docker Image Tag
If using Docker Hub, list available image tags:
```bash
# Using Docker CLI
docker images smartcampus

# Pull specific commit SHA image
docker pull <DOCKERHUB_USERNAME>/smartcampus:<COMMIT_SHA>
```

---

## 3. Git Rollback Procedures

### Scenario A: Reverting a Defective Commit (Recommended)
This approach preserves commit history and creates a clean revert commit.

```bash
# 1. Identify the bad commit hash
git log -n 5 --oneline

# 2. Revert the specific commit without rewriting history
git revert <BAD_COMMIT_HASH> -m 1

# 3. Verify that all tests pass
pytest -v

# 4. Push the revert commit
git push origin main
```

### Scenario B: Restoring Code to a Previous Release Tag
If a full rollback to a known tag (e.g., `v0.1.0`) is required:

```bash
# 1. Create a rollback hotfix branch from the tag
git checkout -b hotfix/rollback-to-v0.1.0 tags/v0.1.0

# 2. Verify application health locally
pytest -v
python smoke_test.py

# 3. Merge hotfix back into main via Pull Request
```

---

## 4. Docker Rollback Procedures

### A. Rolling Back with Docker Compose
If a newly deployed container fails, roll back to the previously built image:

```bash
# 1. Stop the failing container
docker-compose down

# 2. Update docker-compose.yml to reference the previous known good image tag:
# image: <DOCKERHUB_USERNAME>/smartcampus:v0.1.0

# 3. Start container with the stable image
docker-compose up -d

# 4. Verify container health
docker-compose ps
curl -f http://localhost:8501/_stcore/health
```

### B. Standalone Docker Container Rollback
```bash
# 1. Stop and remove the current failing container
docker stop smartcampus-service
docker rm smartcampus-service

# 2. Run previous container tag using existing volumes
docker run -d \
  --name smartcampus-service \
  -p 8501:8501 \
  -v chroma_data:/app/chroma_db \
  -v $(pwd)/data:/app/data:ro \
  -e GROQ_API_KEY="your-api-key" \
  <DOCKERHUB_USERNAME>/smartcampus:<PREVIOUS_COMMIT_SHA>

# 3. Probe health endpoint
curl -f http://localhost:8501/_stcore/health
```

---

## 5. Protecting ChromaDB & Knowledge Base Data During Rollback

> [!CAUTION]
> Never delete `./chroma_db` or the Docker volume `chroma_data` during rollback. The vector store contains indexed embeddings that are backwards-compatible.

### Data Protection Rules:
1. **Read-Only Document Mounts:** The `./data` directory is mounted with `:ro` (read-only) in `docker-compose.yml` to prevent accidental file deletion or modification by container processes.
2. **Backing Up ChromaDB Before Rollback:**
   ```bash
   # Create a timestamped backup of the ChromaDB directory
   cp -r chroma_db/ chroma_db_backup_$(date +%Y%m%d_%H%M%S)/
   ```
3. **Restoring ChromaDB from Backup:**
   ```bash
   # Restore only if database corruption is confirmed
   rm -rf chroma_db/
   cp -r chroma_db_backup_<TIMESTAMP>/ chroma_db/
   ```

---

## 6. Post-Rollback Verification Checklist

After executing any rollback procedure, run the following verification steps:

- [ ] **HTTP Health:** `curl -f http://localhost:8501/_stcore/health` returns `ok`.
- [ ] **Automated Smoke Test:** Run `python smoke_test.py --url http://localhost:8501`.
- [ ] **Document Count:** Verify ChromaDB collection count using `python show_chroma.py`.
- [ ] **Direct Query Verification:** Verify an attendance question returns 75% rule.
- [ ] **Guardrail Verification:** Verify out-of-domain queries return the refusal string.
- [ ] **Log Inspection:** Check container logs (`docker logs smartcampus-service`) for any unhandled exceptions.

