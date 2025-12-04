# GitHub Secrets Setup Guide

To enable automated scraping via GitHub Actions, you need to set up three secrets in your repository.

---

## Step-by-Step Instructions

### 1. Navigate to Repository Settings

1. Go to your GitHub repository: `https://github.com/your-username/Miles`
2. Click **"Settings"** tab (top right)
3. In left sidebar, click **"Secrets and variables"** → **"Actions"**

---

### 2. Add Secrets

Click **"New repository secret"** button for each of these:

#### Secret 1: `SUPABASE_URL`

```
Name: SUPABASE_URL
Value: https://emsychdazifoqcsuurta.supabase.co
```

#### Secret 2: `SUPABASE_SERVICE_KEY`

```
Name: SUPABASE_SERVICE_KEY
Value: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtc3ljaGRhemlmb3Fjc3V1cnRhIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NDc4OTAwNCwiZXhwIjoyMDgwMzY1MDA0fQ.jYaL7IR6v3bXo0EnIvX5ytDnuKPV0AU2lgnK8Yb_YUs
```

⚠️ **Important**: This is your service role key - keep it secret!

#### Secret 3: `INFERENCE_API_URL`

**For Phase 2 (Mock API)**:
```
Name: INFERENCE_API_URL
Value: http://localhost:3000/api/predict
```

**For Phase 3+ (Real ML Model)**:
```
Name: INFERENCE_API_URL
Value: https://your-username-miles-inference.hf.space/predict
```

Note: You'll update this in Phase 3 when you deploy to HuggingFace Spaces

---

## Verification

### Test the Workflow

1. Go to **"Actions"** tab in your repository
2. Select **"YouTube Scraper CRON"** workflow
3. Click **"Run workflow"** dropdown (right side)
4. Click green **"Run workflow"** button
5. Wait ~3-5 minutes
6. Check if the workflow succeeds (green checkmark)

### Expected Behavior

✅ **Success**: Green checkmark, scraper runs, videos added to database
❌ **Failure**: Red X, check logs for errors

---

## Common Issues

### Issue 1: "Secret not found"
- **Solution**: Ensure secret names are EXACTLY as shown (case-sensitive)
- Check for typos: `SUPABASE_URL` not `supabase_url`

### Issue 2: "Authentication failed"
- **Solution**: Verify you're using the **service_role** key, not the anon key
- Go to Supabase → Project Settings → API → Copy service_role key

### Issue 3: "Connection refused"
- **Solution**: For Phase 2, this is expected (localhost not accessible from GitHub)
- The scraper will still work, only inference will fail
- Fix in Phase 3 by deploying real inference service

---

## Security Notes

🔒 **NEVER** commit these secrets to your repository
🔒 **NEVER** share your service_role key publicly
🔒 GitHub Secrets are encrypted and only accessible to workflows
🔒 You can update secrets anytime without affecting the workflow file

---

## Updating Secrets

To update a secret (e.g., after Phase 3):

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click the secret name
3. Click **"Update secret"**
4. Paste new value
5. Click **"Update secret"** button

---

## Workflow Schedule

The scraper runs automatically:
- **Every 6 hours**: 12:00 AM, 6:00 AM, 12:00 PM, 6:00 PM (UTC)
- **Manual trigger**: Anytime via Actions tab

To change schedule, edit `.github/workflows/scraper-cron.yml`:
```yaml
schedule:
  - cron: '0 */6 * * *'  # Every 6 hours
  # Examples:
  # - cron: '0 */12 * * *'  # Every 12 hours
  # - cron: '0 0 * * *'     # Daily at midnight
  # - cron: '0 8,20 * * *'  # 8 AM and 8 PM daily
```

---

## Next Steps

After setting up secrets:
1. ✅ Test manual workflow trigger
2. ✅ Verify videos appear in Supabase
3. ✅ Check workflow logs if issues occur
4. ✅ Update `INFERENCE_API_URL` after Phase 3

---

**That's it!** Your automated scraper is now ready to run.
