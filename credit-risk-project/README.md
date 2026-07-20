# Customer Credit Risk Predictor

Full-stack ML project: udhaar/credit default risk predict karta hai customer history se.

## Structure
```
credit-risk-project/
├── backend/
│   ├── train_model.py       # Model training script
│   ├── app.py                # FastAPI serving API (port 8001)
│   ├── requirements.txt
│   ├── customer_data.csv     # (apna real data yahan daalein)
│   └── credit_risk_model.pkl # (training ke baad generate hoga)
└── frontend/                 # Complete Vite React project
    └── src/
        ├── CreditRiskDashboard.jsx
        └── CreditRiskDashboard.css
```

## Step 1: Apna Data Tayyar Karein

`backend/customer_data.csv` banayein is format mein (General Store/Udhaar system se export kar sakte hain):

```csv
customer_id,total_purchases,total_credit_amount,total_paid_amount,avg_days_to_pay,max_days_overdue,defaulted
CUST001,45,25000,25000,5,0,0
CUST002,12,8000,3000,45,60,1
...
```

**Column meanings:**
- `total_purchases` — customer ne kitni dafa khareeda (count)
- `total_credit_amount` — total kitna udhaar diya gaya (Rs)
- `total_paid_amount` — total kitna wapas mila (Rs)
- `avg_days_to_pay` — average kitne din mein payment karta hai
- `max_days_overdue` — sab se zyada kitne din late hua
- `defaulted` — **TARGET column**: 0 = time pe/theek payment karta hai, 1 = default/bohat late karta hai

Agar aapke real data mein columns ke naam ya structure different hai, mujhe bata dein — training script adjust kar dunga.

## Step 2: Model Train Karein

```bash
cd backend
pip install -r requirements.txt
python train_model.py
```

Agar `customer_data.csv` nahi milta, script khud dummy data (500 customers) generate kar dega taake pipeline test ho sake.

## Step 3: Backend Run Karein

```bash
uvicorn app:app --reload --port 8001
```

Test karein: http://localhost:8001/customers

## Step 4: Frontend Run Karein

```bash
cd frontend
npm install
npm run dev
```

Browser mein `http://localhost:5173` khulega.

## Features
- ✅ **Logistic Regression** (scikit-learn) with feature scaling — simple, fast, explainable
- ✅ FastAPI REST endpoints: `/predict-risk` (single), `/customers` (bulk list)
- ✅ React dashboard: risk summary cards (High/Medium/Low), filterable table, risk bars
- ✅ Dark glassmorphism UI matching your standard theme
- ✅ CORS enabled, environment-variable based API URL (deployment-ready)

## Kyun Logistic Regression?
- Random Forest already Sales Forecast project mein use ho chuka hai — variety ke liye ye simpler, linear model hai
- Coefficients se pata chalta hai konsa factor sabse zyada risk badhata hai (e.g. `max_days_overdue` sabse strong predictor nikla)
- Training ke waqt terminal mein feature coefficients print hote hain — transparency ke liye acha hai

## Deployment
Same tareeqa jo Sales Forecast project ke liye use kiya:
- **Backend** → Render.com (Root Directory: `backend`, Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`)
- **Frontend** → Vercel.com (Root Directory: `frontend`, Environment Variable: `VITE_API_URL` = Render backend URL)

## Next Steps (Submission ke liye)
- [ ] Real customer data CSV tayyar (`customer_data.csv`)
- [ ] Model train ho gaya (`credit_risk_model.pkl` generated)
- [ ] Backend deployed
- [ ] Frontend deployed
- [ ] UI dashboard mein integrated

## 🚀 GitHub Pages pe Deploy Karna (Live Link)

Ye project ab **fully static/client-side** hai — Logistic Regression ke coefficients aur scaler JS mein embed hain, koi Python backend zaroori nahi.

### Steps:
1. Naya GitHub repo banayein (e.g. `credit-risk-project`)
2. `frontend/vite.config.js` mein `base: '/credit-risk-project/'` confirm karein (repo name se match hona chahiye)
3. Push karein:
   ```bash
   git init && git add . && git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/credit-risk-project.git
   git push -u origin main
   ```
4. Settings → Pages → Source: **GitHub Actions**
5. Live link: `https://YOUR-USERNAME.github.io/credit-risk-project/`

Backend files reference/documentation ke liye rakhe gaye hain — live site poori tarah browser mein chalti hai.
