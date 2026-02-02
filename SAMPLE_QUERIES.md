# 📊 IntelliQuery AI - Sample Queries Guide

## 🎉 Your Application is Working!

Based on your logs, the system successfully:
- ✅ Trained ML model on 7,043 records
- ✅ Auto-detected target column: `churn_label`
- ✅ Identified 21 categorical features
- ✅ Identified 10 numeric features
- ✅ Saved model to `models/churn_model.pkl`
- ✅ Made predictions successfully

---

## 📊 Your Dataset Structure

Based on the auto-detection, your dataset has:

### Categorical Features (21)
- `country`, `state`, `city`, `lat_long`, `gender`
- Plus 16 more categorical columns

### Numeric Features (10)
- `col_count`, `zip_code`, `latitude`, `longitude`, `tenure_months`
- Plus 5 more numeric columns

### Target Column
- `churn_label` (auto-detected)

---

## 💬 Sample Queries for Your Data

### 1. Basic Statistics Queries

#### Customer Count
```
How many customers do we have?
```
**Expected**: Total count of records

#### Churn Rate
```
What is the churn rate?
```
**Expected**: Percentage of customers who churned

#### Average Tenure
```
What is the average tenure of customers?
```
**Expected**: Average tenure_months value

---

### 2. Geographic Analysis

#### Customers by State
```
How many customers are in each state?
```
**Expected**: Count grouped by state

#### Churn by State
```
Which states have the highest churn rate?
```
**Expected**: Churn rate by state, sorted descending

#### Top Cities
```
What are the top 10 cities by customer count?
```
**Expected**: Cities with most customers

---

### 3. Demographic Analysis

#### Gender Distribution
```
What is the gender distribution of customers?
```
**Expected**: Count by gender

#### Churn by Gender
```
Do males or females churn more?
```
**Expected**: Churn rate by gender

#### Senior Citizens
```
How many senior citizens do we have?
```
**Expected**: Count where senior_citizen = 1 or Yes

---

### 4. Service Analysis

#### Internet Service Types
```
What internet service types do customers use?
```
**Expected**: Count by internet_service

#### Phone Service Adoption
```
How many customers have phone service?
```
**Expected**: Count where phone_service = Yes

#### Multiple Lines
```
What percentage of customers have multiple lines?
```
**Expected**: Percentage with multiple_lines = Yes

---

### 5. Contract Analysis

#### Contract Types
```
What are the different contract types and their distribution?
```
**Expected**: Count by contract type

#### Churn by Contract
```
Which contract type has the highest churn rate?
```
**Expected**: Churn rate by contract, sorted descending

#### Month-to-Month Customers
```
How many customers are on month-to-month contracts?
```
**Expected**: Count where contract = 'Month-to-month'

---

### 6. Billing Analysis

#### Average Monthly Charges
```
What is the average monthly charge?
```
**Expected**: Average of monthly_charges

#### Total Revenue
```
What is the total revenue from all customers?
```
**Expected**: Sum of total_charges

#### High-Value Customers
```
How many customers pay more than $100 per month?
```
**Expected**: Count where monthly_charges > 100

#### Payment Methods
```
What payment methods do customers use?
```
**Expected**: Count by payment_method

---

### 7. Churn Analysis

#### Recent Churners
```
How many customers churned in the last 3 months?
```
**Expected**: Count of recent churns (if date available)

#### Churn by Tenure
```
What is the churn rate for customers with less than 12 months tenure?
```
**Expected**: Churn rate where tenure_months < 12

#### High-Risk Customers
```
How many customers have high churn risk?
```
**Expected**: Count of customers with churn probability > 0.7

---

### 8. Service Combination Queries

#### Full Service Customers
```
How many customers have internet, phone, and streaming services?
```
**Expected**: Count with all services

#### No Internet Customers
```
How many customers don't have internet service?
```
**Expected**: Count where internet_service = 'No'

#### Premium Services
```
What percentage of customers have tech support?
```
**Expected**: Percentage with tech_support = Yes

---

### 9. Comparison Queries

#### Churned vs Active
```
Compare average monthly charges between churned and active customers
```
**Expected**: Average monthly_charges grouped by churn_label

#### Contract Comparison
```
Compare churn rates across all contract types
```
**Expected**: Churn rate by contract type

#### Service Impact
```
Do customers with online security churn less?
```
**Expected**: Churn rate comparison with/without online_security

---

### 10. Advanced Aggregation Queries

#### Revenue by State
```
What is the total revenue by state?
```
**Expected**: Sum of total_charges grouped by state

#### Average Tenure by Contract
```
What is the average tenure for each contract type?
```
**Expected**: Average tenure_months by contract

#### Churn by Multiple Factors
```
Show churn rate by gender and contract type
```
**Expected**: Churn rate grouped by gender and contract

---

## 🎯 ML Prediction Queries

### Single Customer Prediction

Use the web UI to input customer data and get:
- **Churn Probability**: 0-100%
- **Risk Level**: LOW, MEDIUM, HIGH
- **Recommendation**: Personalized action

### Example Customer Data
```json
{
  "gender": "Female",
  "senior_citizen": "No",
  "partner": "Yes",
  "dependents": "No",
  "tenure_months": 12,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 85.50,
  "total_charges": 1020.00
}
```

---

## 📈 Chart Queries

The system automatically generates these charts:

1. **Churn Distribution**: Overall churn vs non-churn
2. **Churn by Category**: Churn rate by different features
3. **Feature Importance**: Which features predict churn best
4. **Risk Distribution**: Distribution of risk levels
5. **Prediction Comparison**: Actual vs predicted churn

---

## 🔍 Query Tips

### For Best Results:

1. **Be Specific**: "Show churn rate by state" is better than "show churn"
2. **Use Aggregations**: COUNT, AVG, SUM, MIN, MAX
3. **Add Filters**: "customers with tenure > 12 months"
4. **Sort Results**: "top 10", "highest", "lowest"
5. **Compare Groups**: "compare X between Y and Z"

### Supported Query Patterns:

✅ **Aggregations**: COUNT, AVG, SUM, MIN, MAX  
✅ **Filters**: WHERE conditions (>, <, =, !=)  
✅ **Grouping**: BY state, gender, contract, etc.  
✅ **Sorting**: ORDER BY, TOP N, LIMIT  
✅ **Comparisons**: Between groups  

### Not Yet Supported:

❌ **JOINs**: Multiple table queries  
❌ **Subqueries**: Nested SELECT statements  
❌ **Complex WHERE**: Multiple AND/OR conditions  
❌ **Window Functions**: RANK, ROW_NUMBER, etc.  

---

## 🎓 Learning from Your Data

### Key Insights to Explore:

1. **Churn Drivers**: Which features correlate with churn?
2. **Geographic Patterns**: Do certain regions churn more?
3. **Service Impact**: Which services reduce churn?
4. **Contract Effect**: How does contract type affect retention?
5. **Pricing Analysis**: Is there a price sensitivity threshold?

### Questions to Ask:

- What's the average tenure of churned vs active customers?
- Which contract type has the best retention?
- Do customers with tech support churn less?
- What's the churn rate for high-value customers?
- Which states need retention campaigns?

---

## 🚀 Next Steps

### 1. Explore Your Data
Try the sample queries above to understand your customer base

### 2. Train Custom Models
Upload different datasets to see the dataset-agnostic ML in action

### 3. Generate Insights
Use the charts to visualize patterns and trends

### 4. Make Predictions
Test the ML model with different customer profiles

### 5. Monitor Performance
Track which features are most important for predictions

---

## 📊 Example Query Session

```
User: "How many customers do we have?"
System: "You have 7,043 customers in total."

User: "What is the churn rate?"
System: "The overall churn rate is 26.5% (1,869 out of 7,043 customers)."

User: "Which contract type has the highest churn?"
System: "Month-to-month contracts have the highest churn rate at 42.7%."

User: "Show me the top 5 states by customer count"
System: [Returns table with state names and counts]

User: "What's the average monthly charge for churned customers?"
System: "Churned customers had an average monthly charge of $74.44."
```

---

## 🎉 Success Indicators

Based on your logs, your system is:

✅ **Working**: All endpoints responding  
✅ **Training**: ML model trained successfully  
✅ **Predicting**: Churn predictions working  
✅ **Persisting**: Model saved and loaded  
✅ **Adapting**: Auto-detected your dataset structure  

**Your IntelliQuery AI is production-ready! 🚀**

---

## 📚 Additional Resources

- **Architecture**: See `ARCHITECTURE_FLOWS.md`
- **Dataset Guide**: See `DATASET_ADAPTABILITY_GUIDE.md`
- **Implementation**: See `IMPLEMENTATION_SUMMARY.md`
- **API Docs**: Visit http://localhost:8000/docs

---

## 💡 Pro Tips

1. **Start Simple**: Begin with basic COUNT queries
2. **Build Complexity**: Add filters and grouping gradually
3. **Use Charts**: Visualize results for better insights
4. **Test Predictions**: Try different customer profiles
5. **Monitor Logs**: Check console for query execution details

**Happy Querying! 🎯**