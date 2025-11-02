# Machine Learning Model Explorer

An interactive **Streamlit web app** for exploring Machine Learning models. It supports both **Supervised** and **Unsupervised** learning, automatic preprocessing, and visual evaluation.

## Features
- Upload and preview CSV datasets
- Auto handle missing values, encoding, and scaling
- Model selection and hyperparameter tuning
- Performance metrics and plots (confusion matrix, residuals, clusters)

### Supervised
- Decision Tree, Random Forest, SVM (Classification & Regression)

### Unsupervised
- KMeans, Agglomerative, DBSCAN

## Run Locally
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Access at: http://localhost:8051/
Global access at: http://3.90.44.142:8501/

## Deploy on AWS EC2
- Launch Ubuntu instance
- `sudo apt install python3-pip python3-venv git`
- Clone repo and install dependencies
- Run: `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`
- Add inbound rule for port 8501 in EC2 Security Group

## Author
**Ahmed Faraz Shaikh**
