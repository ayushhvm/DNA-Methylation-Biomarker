import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io

st.set_page_config(page_title="DNA Methylation Prediction", layout="wide")

st.title("DNA Methylation State Prediction")
st.write("Upload your training and test datasets to predict DNA methylation states using multiple machine learning algorithms.")

# Sidebar for file uploads
st.sidebar.header("Data Upload")
train_file = st.sidebar.file_uploader("Upload Training Data (CSV)", type=['csv'])
test_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=['csv'])

# Well-trained models upload
st.sidebar.header("Well Trained Models")
st.sidebar.write("Upload pre-trained model files:")
logistic_model_file = st.sidebar.file_uploader("Logistic Regression Model", type=['pkl'], key="logistic")
random_forest_model_file = st.sidebar.file_uploader("Random Forest Model", type=['pkl'], key="rf")
decision_tree_model_file = st.sidebar.file_uploader("Decision Tree Model", type=['pkl'], key="dt")
knn_model_file = st.sidebar.file_uploader("KNN Model", type=['pkl'], key="knn")

# Model parameters
st.sidebar.header("Model Parameters")
m = st.sidebar.slider("Sequence window size (m)", min_value=100, max_value=1000, value=500, step=50)
rf_max_depth = st.sidebar.slider("Random Forest Max Depth", min_value=1, max_value=10, value=2)
gb_n_estimators = st.sidebar.slider("Gradient Boosting N Estimators", min_value=50, max_value=200, value=100, step=25)
knn_neighbors = st.sidebar.slider("KNN Neighbors", min_value=1, max_value=10, value=3)

def process_data(df, m_value, is_training=True):
    """Process the dataframe to extract features"""
    if is_training:
        df['CG'] = df.seq.apply(lambda x: x[1000-m_value:1000+m_value].count('CG'))
        df['TG'] = df.seq.apply(lambda x: x[1000-m_value:1000+m_value].count('TG'))
        df['CA'] = df.seq.apply(lambda x: x[1000-m_value:1000+m_value].count('CA'))
    else:
        df['CG'] = df.seq.apply(lambda x: x[500:1500].count('CG'))
        df['TG'] = df.seq.apply(lambda x: x[500:1500].count('TG'))
        df['CA'] = df.seq.apply(lambda x: x[500:1500].count('CA'))
    
    df['mutation'] = (df.TG + df.CA) / (2 * df.CG)
    df = pd.get_dummies(df, columns=['Regulatory_Feature_Group'])
    
    return df

def calculate_model_agreement(predictions_dict):
    """Calculate pairwise agreement (concordance) between models"""
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    # Create agreement matrix
    agreement_matrix = np.zeros((n_models, n_models))
    agreement_data = []
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i != j:
                # Calculate concordance (agreement percentage)
                concordance = np.mean(predictions_dict[model1] == predictions_dict[model2]) * 100
                agreement_matrix[i, j] = concordance
                
                if i < j:  # Only store unique pairs
                    agreement_data.append({
                        'Model 1': model1,
                        'Model 2': model2,
                        'Concordance (%)': concordance,
                        'Discordance (%)': 100 - concordance
                    })
            else:
                agreement_matrix[i, j] = 100  # Perfect agreement with itself
    
    return agreement_matrix, agreement_data, model_names

def calculate_comprehensive_metrics(y_true, y_pred, y_prob=None, model_name="Model"):
    """Calculate comprehensive performance metrics"""
    metrics_dict = {
        'Model': model_name,
        'Accuracy (%)': accuracy_score(y_true, y_pred) * 100,
        'Precision (%)': precision_score(y_true, y_pred, average='weighted') * 100,
        'Recall (%)': recall_score(y_true, y_pred, average='weighted') * 100,
        'F1-Score (%)': f1_score(y_true, y_pred, average='weighted') * 100,
        'Specificity (%)': None,  # Will calculate below
        'Sensitivity (%)': None   # Will calculate below
    }
    
    # Calculate confusion matrix for specificity and sensitivity
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        metrics_dict['Specificity (%)'] = specificity
        metrics_dict['Sensitivity (%)'] = sensitivity
    
    if y_prob is not None:
        try:
            auc_score = roc_auc_score(y_true, y_prob) * 100
            metrics_dict['AUC (%)'] = auc_score
        except:
            metrics_dict['AUC (%)'] = None
    
    return metrics_dict

def plot_model_comparison_charts(predictions_dict, probabilities_dict=None, y_true=None):
    """Create comprehensive comparison visualizations"""
    
    # 1. Agreement Heatmap
    agreement_matrix, agreement_data, model_names = calculate_model_agreement(predictions_dict)
    
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(agreement_matrix, annot=True, cmap='Blues', fmt='.1f', 
                xticklabels=model_names, yticklabels=model_names, ax=ax1)
    ax1.set_title('Model Concordance Matrix (%)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Models')
    plt.tight_layout()
    
    # 2. Pairwise Agreement Bar Chart
    if agreement_data:
        agreement_df = pd.DataFrame(agreement_data)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(agreement_df))
        bars1 = ax2.bar(x_pos, agreement_df['Concordance (%)'], 
                       color='lightblue', label='Concordance', alpha=0.8)
        bars2 = ax2.bar(x_pos, agreement_df['Discordance (%)'], 
                       bottom=agreement_df['Concordance (%)'], 
                       color='lightcoral', label='Discordance', alpha=0.8)
        
        ax2.set_xlabel('Model Pairs')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Pairwise Model Agreement Analysis', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{row['Model 1']}\nvs\n{row['Model 2']}" 
                            for _, row in agreement_df.iterrows()], rotation=45)
        ax2.legend()
        ax2.set_ylim(0, 100)
        
        # Add percentage labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax2.text(bar1.get_x() + bar1.get_width()/2., height1/2,
                    f'{height1:.1f}%', ha='center', va='center', fontweight='bold')
            ax2.text(bar2.get_x() + bar2.get_width()/2., height1 + height2/2,
                    f'{height2:.1f}%', ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
    else:
        fig2 = None
    
    # 3. Prediction Distribution Comparison
    fig3, axes3 = plt.subplots(2, 2, figsize=(15, 10))
    axes3 = axes3.ravel()
    
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        if i < 4:  # Maximum 4 subplots
            pred_counts = pd.Series(predictions).value_counts().sort_index()
            colors = ['lightcoral', 'lightblue']
            
            axes3[i].pie(pred_counts.values, labels=[f'Class {int(k)}' for k in pred_counts.index], 
                        autopct='%1.1f%%', colors=colors, startangle=90)
            axes3[i].set_title(f'{model_name}\nPrediction Distribution', fontweight='bold')
    
    # Hide unused subplots
    for i in range(len(predictions_dict), 4):
        axes3[i].axis('off')
    
    plt.tight_layout()
    
    # 4. ROC Curves Comparison (if probabilities available)
    fig4 = None
    if probabilities_dict and y_true is not None:
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (model_name, probs) in enumerate(probabilities_dict.items()):
            try:
                if probs.ndim > 1 and probs.shape[1] > 1:
                    fpr, tpr, _ = metrics.roc_curve(y_true, probs[:, 1])
                else:
                    fpr, tpr, _ = metrics.roc_curve(y_true, probs)
                
                auc_score = roc_auc_score(y_true, probs[:, 1] if probs.ndim > 1 and probs.shape[1] > 1 else probs)
                ax4.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                        label=f'{model_name} (AUC = {auc_score:.3f})')
            except Exception as e:
                continue
        
        ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random Classifier')
        ax4.set_xlim([0.0, 1.0])
        ax4.set_ylim([0.0, 1.05])
        ax4.set_xlabel('False Positive Rate (1 - Specificity)')
        ax4.set_ylabel('True Positive Rate (Sensitivity)')
        ax4.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax4.legend(loc="lower right")
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
    
    return fig1, fig2, fig3, fig4, agreement_data

def load_pretrained_models():
    """Load pre-trained models from uploaded files"""
    pretrained_models = {}
    
    model_files = {
        'Logistic Regression': logistic_model_file,
        'Random Forest': random_forest_model_file,
        'Decision Tree': decision_tree_model_file,
        'KNN': knn_model_file
    }
    
    for model_name, model_file in model_files.items():
        if model_file is not None:
            try:
                model = joblib.load(model_file)
                pretrained_models[model_name] = model
            except Exception as e:
                st.error(f"Error loading {model_name} model: {str(e)}")
    
    return pretrained_models

def train_models(X, y, rf_depth, gb_est, knn_n):
    """Train all models and return them with their performance metrics"""
    models = {}
    results = {}
    
    # Logistic Regression
    clf = LogisticRegression(random_state=0).fit(X, y)
    prob = clf.predict_proba(X)
    auc = roc_auc_score(y, prob[:, 1])
    pred = clf.predict(X)
    report = classification_report(y, pred, output_dict=True)
    
    models['Logistic Regression'] = clf
    results['Logistic Regression'] = {'AUC': auc, 'Report': report, 'Probabilities': prob}
    
    # Random Forest
    clf1 = RandomForestClassifier(max_depth=rf_depth, random_state=0).fit(X, y)
    prob1 = clf1.predict_proba(X)
    auc1 = roc_auc_score(y, prob1[:, 1])
    pred1 = clf1.predict(X)
    report1 = classification_report(y, pred1, output_dict=True)
    
    models['Random Forest'] = clf1
    results['Random Forest'] = {'AUC': auc1, 'Report': report1, 'Probabilities': prob1}
    
    # Gradient Boosting
    clf2 = GradientBoostingClassifier(n_estimators=gb_est, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
    prob2 = clf2.predict_proba(X)
    auc2 = roc_auc_score(y, prob2[:, 1])
    pred2 = clf2.predict(X)
    report2 = classification_report(y, pred2, output_dict=True)
    
    models['Gradient Boosting'] = clf2
    results['Gradient Boosting'] = {'AUC': auc2, 'Report': report2, 'Probabilities': prob2}
    
    # KNN
    clf3 = KNeighborsClassifier(n_neighbors=knn_n).fit(X, y)
    prob3 = clf3.predict_proba(X)
    auc3 = roc_auc_score(y, prob3[:, 1])
    pred3 = clf3.predict(X)
    report3 = classification_report(y, pred3, output_dict=True)
    
    models['KNN'] = clf3
    results['KNN'] = {'AUC': auc3, 'Report': report3, 'Probabilities': prob3}
    
    return models, results

# Load pre-trained models if available
pretrained_models = load_pretrained_models()

if pretrained_models:
    st.header("Well Trained Models Available")
    st.success(f"Loaded {len(pretrained_models)} pre-trained models: {', '.join(pretrained_models.keys())}")
    
    # Store pretrained models in session state
    st.session_state.pretrained_models = pretrained_models
    st.session_state.pretrained_feature_cols = ['mutation', 'Regulatory_Feature_Group_Promoter_Associated']

if train_file is not None:
    # Load and process training data
    df_train = pd.read_csv(train_file, index_col=0)
    
    st.header("Training Data Overview")
    st.write(f"Dataset shape: {df_train.shape}")
    st.write("First few rows:")
    st.dataframe(df_train.head())
    
    # Show value counts for categorical features
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Regulatory Feature Group Distribution")
        reg_counts = df_train['Regulatory_Feature_Group'].value_counts()
        st.write(reg_counts)
    
    with col2:
        st.subheader("CpG Island Relation Distribution")
        cpg_counts = df_train['Relation_to_UCSC_CpG_Island'].value_counts()
        st.write(cpg_counts)
    
    # Process training data
    df_processed = process_data(df_train.copy(), m, is_training=True)
    
    # Prepare features
    feature_cols = ['mutation', 'Regulatory_Feature_Group_Promoter_Associated']
    if 'Regulatory_Feature_Group_Promoter_Associated' not in df_processed.columns:
        df_processed['Regulatory_Feature_Group_Promoter_Associated'] = 0
    
    X = df_processed[feature_cols]
    y = df_processed['Beta']
    
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            models, results = train_models(X, y.values, rf_max_depth, gb_n_estimators, knn_neighbors)
        
        st.success("Models trained successfully!")
        
        # Display results
        st.header("Model Performance")
        
        # Create performance comparison table
        performance_data = []
        for model_name, result in results.items():
            performance_data.append({
                'Model': model_name,
                'AUC Score': f"{result['AUC']:.4f}",
                'Precision (Class 0)': f"{result['Report']['0']['precision']:.4f}",
                'Recall (Class 0)': f"{result['Report']['0']['recall']:.4f}",
                'Precision (Class 1)': f"{result['Report']['1']['precision']:.4f}",
                'Recall (Class 1)': f"{result['Report']['1']['recall']:.4f}",
                'F1-Score': f"{result['Report']['macro avg']['f1-score']:.4f}"
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df)
        
        # Plot ROC curves
        st.header("ROC Curves")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, (model_name, model) in enumerate(models.items()):
            fpr, tpr, _ = metrics.roc_curve(y, results[model_name]['Probabilities'][:, 1])
            auc_score = results[model_name]['AUC']
            
            axes[i].plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})')
            axes[i].plot([0, 1], [0, 1], 'k--')
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'{model_name}')
            axes[i].legend(loc="lower right")
            axes[i].grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Store models in session state
        st.session_state.models = models
        st.session_state.feature_cols = feature_cols
        
        # Download models
        st.header("Download Trained Models")
        
        for model_name, model in models.items():
            model_bytes = io.BytesIO()
            joblib.dump(model, model_bytes)
            model_bytes.seek(0)
            
            st.download_button(
                label=f"Download {model_name} Model",
                data=model_bytes.getvalue(),
                file_name=f"{model_name.lower().replace(' ', '_')}_model.pkl",
                mime="application/octet-stream"
            )

# Test data prediction
if test_file is not None and ('models' in st.session_state or 'pretrained_models' in st.session_state):
    st.header("Test Data Prediction")
    
    # Load test data
    df_test = pd.read_csv(test_file, index_col=0)
    st.write(f"Test dataset shape: {df_test.shape}")
    st.write("First few rows:")
    st.dataframe(df_test.head())
    
    # Process test data
    df_test_processed = process_data(df_test.copy(), m, is_training=False)
    
    # Determine available models and feature columns
    available_models = {}
    feature_cols = ['mutation', 'Regulatory_Feature_Group_Promoter_Associated']
    
    if 'models' in st.session_state:
        available_models.update(st.session_state.models)
        feature_cols = st.session_state.feature_cols
    
    if 'pretrained_models' in st.session_state:
        # Add prefix to distinguish pretrained models
        for name, model in st.session_state.pretrained_models.items():
            available_models[f"Pre-trained {name}"] = model
    
    # Ensure all required columns exist
    for col in feature_cols:
        if col not in df_test_processed.columns:
            df_test_processed[col] = 0
    
    X_test = df_test_processed[feature_cols]
    
    # Model selection for prediction
    st.subheader("Model Selection")
    
    # Separate newly trained and pre-trained models
    newly_trained = [name for name in available_models.keys() if not name.startswith("Pre-trained")]
    pretrained = [name for name in available_models.keys() if name.startswith("Pre-trained")]
    
    if newly_trained:
        st.write("**Newly Trained Models:**")
        for model in newly_trained:
            st.write(f"- {model}")
    
    if pretrained:
        st.write("**Pre-trained Models:**")
        for model in pretrained:
            st.write(f"- {model}")
    
    selected_model = st.selectbox("Select model for prediction:", list(available_models.keys()))
    
    if st.button("Make Predictions"):
        model = available_models[selected_model]
        
        try:
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            
            # Create results dataframe
            results_df = df_test.copy()
            results_df['Beta'] = predictions
            results_df['Probability_Class_0'] = probabilities[:, 0]
            results_df['Probability_Class_1'] = probabilities[:, 1]
            
            st.success(f"Predictions made using {selected_model}")
            st.dataframe(results_df[['Beta', 'Probability_Class_0', 'Probability_Class_1']].head(10))
            
            # Download predictions
            csv_buffer = io.StringIO()
            results_df[['Beta']].to_csv(csv_buffer)
            
            st.download_button(
                label="Download Predictions (solution.csv)",
                data=csv_buffer.getvalue(),
                file_name="solution.csv",
                mime="text/csv"
            )
            
            # Show prediction distribution
            st.subheader("Prediction Distribution")
            pred_counts = pd.Series(predictions).value_counts().sort_index()
            st.bar_chart(pred_counts)
            
            # Show prediction statistics
            st.subheader("Prediction Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", len(predictions))
            with col2:
                st.metric("Class 0 (Unmethylated)", int(np.sum(predictions == 0)))
            with col3:
                st.metric("Class 1 (Methylated)", int(np.sum(predictions == 1)))
                
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            st.write("Please ensure the test data format matches the training data format.")

# Model Comparison Section
if ('pretrained_models' in st.session_state and len(st.session_state.pretrained_models) >= 2) or test_file is not None:
    if test_file is not None and ('models' in st.session_state or 'pretrained_models' in st.session_state):
        st.header("🔍 Comprehensive Model Comparison & Analysis")
        
        # Load test data for comparison
        if 'df_test_processed' not in locals():
            df_test = pd.read_csv(test_file, index_col=0)
            df_test_processed = process_data(df_test.copy(), m, is_training=False)
            
            # Determine available models and feature columns
            available_models = {}
            feature_cols = ['mutation', 'Regulatory_Feature_Group_Promoter_Associated']
            
            if 'models' in st.session_state:
                available_models.update(st.session_state.models)
                feature_cols = st.session_state.feature_cols
            
            if 'pretrained_models' in st.session_state:
                for name, model in st.session_state.pretrained_models.items():
                    available_models[f"Pre-trained {name}"] = model
            
            # Ensure all required columns exist
            for col in feature_cols:
                if col not in df_test_processed.columns:
                    df_test_processed[col] = 0
            
            X_test = df_test_processed[feature_cols]
        
        if st.button("🚀 Run Comprehensive Model Comparison", type="primary"):
            with st.spinner("Running comprehensive analysis..."):
                # Get predictions and probabilities from all models
                all_predictions = {}
                all_probabilities = {}
                all_metrics = []
                
                for model_name, model in available_models.items():
                    try:
                        predictions = model.predict(X_test)
                        probabilities = model.predict_proba(X_test)
                        
                        all_predictions[model_name] = predictions
                        all_probabilities[model_name] = probabilities
                        
                        # Calculate metrics (without true labels, we'll show prediction-based metrics)
                        pred_metrics = {
                            'Model': model_name,
                            'Total Predictions': len(predictions),
                            'Class 0 Predictions': int(np.sum(predictions == 0)),
                            'Class 1 Predictions': int(np.sum(predictions == 1)),
                            'Class 0 Percentage (%)': np.mean(predictions == 0) * 100,
                            'Class 1 Percentage (%)': np.mean(predictions == 1) * 100,
                            'Mean Probability Class 0': np.mean(probabilities[:, 0]),
                            'Mean Probability Class 1': np.mean(probabilities[:, 1]),
                            'Prediction Confidence': np.mean(np.max(probabilities, axis=1))
                        }
                        all_metrics.append(pred_metrics)
                        
                    except Exception as e:
                        st.error(f"Error with model {model_name}: {str(e)}")
                        continue
            
            if all_predictions:
                # Display comprehensive metrics table
                st.subheader("📊 Model Performance Summary")
                metrics_df = pd.DataFrame(all_metrics)
                st.dataframe(metrics_df.round(3))
                
                # Download metrics
                csv_buffer = io.StringIO()
                metrics_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Performance Metrics",
                    data=csv_buffer.getvalue(),
                    file_name="model_performance_metrics.csv",
                    mime="text/csv"
                )
                
                # Model Agreement Analysis
                st.subheader("🤝 Model Agreement Analysis (Concordance)")
                
                fig1, fig2, fig3, fig4, agreement_data = plot_model_comparison_charts(
                    all_predictions, all_probabilities
                )
                
                # Display agreement heatmap
                st.pyplot(fig1)
                
                # Display pairwise agreement
                if fig2:
                    st.pyplot(fig2)
                
                # Display agreement statistics
                if agreement_data:
                    st.subheader("📈 Pairwise Concordance Statistics")
                    agreement_df = pd.DataFrame(agreement_data)
                    
                    # Highlight agreement levels
                    def highlight_agreement(val):
                        if val >= 90:
                            return 'background-color: darkgreen; color: white'
                        elif val >= 80:
                            return 'background-color: green; color: white'
                        elif val >= 70:
                            return 'background-color: orange; color: white'
                        else:
                            return 'background-color: red; color: white'
                    
                    styled_df = agreement_df.style.applymap(highlight_agreement, subset=['Concordance (%)'])
                    st.dataframe(styled_df)
                    
                    # Agreement interpretation
                    st.subheader("🎯 Agreement Interpretation")
                    avg_concordance = agreement_df['Concordance (%)'].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Concordance", f"{avg_concordance:.1f}%")
                    with col2:
                        max_concordance = agreement_df['Concordance (%)'].max()
                        best_pair = agreement_df.loc[agreement_df['Concordance (%)'].idxmax()]
                        st.metric("Highest Agreement", f"{max_concordance:.1f}%")
                        st.caption(f"{best_pair['Model 1']} vs {best_pair['Model 2']}")
                    with col3:
                        min_concordance = agreement_df['Concordance (%)'].min()
                        worst_pair = agreement_df.loc[agreement_df['Concordance (%)'].idxmin()]
                        st.metric("Lowest Agreement", f"{min_concordance:.1f}%")
                        st.caption(f"{worst_pair['Model 1']} vs {worst_pair['Model 2']}")
                    
                    # Agreement levels explanation
                    st.info("""
                    **Concordance Interpretation:**
                    - 🟢 **90-100%**: Excellent agreement (models are highly consistent)
                    - 🟡 **80-89%**: Good agreement (models generally agree)
                    - 🟠 **70-79%**: Moderate agreement (some disagreement expected)
                    - 🔴 **<70%**: Poor agreement (models disagree frequently)
                    """)
                
                # Display prediction distributions
                st.subheader("📊 Prediction Distribution Comparison")
                st.pyplot(fig3)
                
                # Display ROC comparison if available
                if fig4:
                    st.subheader("📈 ROC Curves Comparison")
                    st.pyplot(fig4)
                
                # Detailed prediction comparison table
                st.subheader("🔍 Detailed Prediction Comparison")
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame(all_predictions)
                comparison_df.index.name = 'Sample_ID'
                
                # Add consensus column
                comparison_df['Consensus'] = comparison_df.mode(axis=1)[0]
                comparison_df['Agreement_Count'] = comparison_df.iloc[:, :-1].eq(comparison_df.iloc[:, :-1].mode(axis=1)[0], axis=0).sum(axis=1)
                comparison_df['Unanimous'] = comparison_df['Agreement_Count'] == len(all_predictions)
                
                st.write("Sample of prediction comparisons:")
                st.dataframe(comparison_df.head(20))
                
                # Download full comparison
                csv_buffer = io.StringIO()
                comparison_df.to_csv(csv_buffer)
                st.download_button(
                    label="📥 Download Full Prediction Comparison",
                    data=csv_buffer.getvalue(),
                    file_name="model_predictions_comparison.csv",
                    mime="text/csv"
                )
                
                # Consensus statistics
                st.subheader("🎯 Consensus Analysis")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    unanimous_count = comparison_df['Unanimous'].sum()
                    st.metric("Unanimous Predictions", f"{unanimous_count:,}")
                    st.caption(f"{unanimous_count/len(comparison_df)*100:.1f}% of total")
                
                with col2:
                    majority_class_0 = (comparison_df['Consensus'] == 0).sum()
                    st.metric("Consensus Class 0", f"{majority_class_0:,}")
                    st.caption(f"{majority_class_0/len(comparison_df)*100:.1f}% of total")
                
                with col3:
                    majority_class_1 = (comparison_df['Consensus'] == 1).sum()
                    st.metric("Consensus Class 1", f"{majority_class_1:,}")
                    st.caption(f"{majority_class_1/len(comparison_df)*100:.1f}% of total")
                
                with col4:
                    avg_agreement = comparison_df['Agreement_Count'].mean()
                    st.metric("Avg Model Agreement", f"{avg_agreement:.1f}/{len(all_predictions)}")
                    st.caption(f"{avg_agreement/len(all_predictions)*100:.1f}% consensus")

elif test_file is not None and 'models' not in st.session_state and 'pretrained_models' not in st.session_state:
    st.warning("Please either train models using training data or upload pre-trained models before making predictions on test data.")

# Model Performance with Ground Truth (if training data available)
if train_file is not None and 'models' in st.session_state:
    st.header("🎯 Model Performance with Ground Truth")
    
    if st.button("📊 Evaluate Models on Training Data"):
        with st.spinner("Evaluating model performance..."):
            # Use the training data for evaluation
            ground_truth_metrics = []
            
            for model_name, model in st.session_state.models.items():
                try:
                    # Get predictions on training data
                    train_predictions = model.predict(X)
                    train_probabilities = model.predict_proba(X)
                    
                    # Calculate comprehensive metrics
                    metrics_dict = calculate_comprehensive_metrics(
                        y.values, train_predictions, 
                        train_probabilities[:, 1] if train_probabilities.shape[1] > 1 else train_probabilities,
                        model_name
                    )
                    ground_truth_metrics.append(metrics_dict)
                    
                except Exception as e:
                    st.error(f"Error evaluating {model_name}: {str(e)}")
                    continue
            
            if ground_truth_metrics:
                # Display performance metrics with ground truth
                st.subheader("🎯 Accuracy & Performance Metrics")
                gt_metrics_df = pd.DataFrame(ground_truth_metrics)
                
                # Style the dataframe to highlight best performance
                def highlight_best(s):
                    if s.name in ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'AUC (%)', 'Specificity (%)', 'Sensitivity (%)']:
                        is_max = s == s.max()
                        return ['background-color: lightgreen' if v else '' for v in is_max]
                    return ['' for _ in s]
                
                styled_gt_df = gt_metrics_df.style.apply(highlight_best, axis=0)
                st.dataframe(styled_gt_df)
                
                # Performance summary
                st.subheader("🏆 Performance Summary")
                
                # Find best performing model for each metric
                best_models = {}
                for metric in ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'AUC (%)']:
                    if metric in gt_metrics_df.columns:
                        best_idx = gt_metrics_df[metric].idxmax()
                        best_models[metric] = {
                            'model': gt_metrics_df.iloc[best_idx]['Model'],
                            'score': gt_metrics_df.iloc[best_idx][metric]
                        }
                
                cols = st.columns(len(best_models))
                for i, (metric, info) in enumerate(best_models.items()):
                    with cols[i]:
                        st.metric(
                            f"Best {metric.replace(' (%)', '')}",
                            f"{info['score']:.2f}%",
                            delta=None
                        )
                        st.caption(f"Model: {info['model']}")
                
                # Model ranking
                st.subheader("🥇 Overall Model Ranking")
                
                # Calculate average performance score
                score_columns = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']
                available_score_cols = [col for col in score_columns if col in gt_metrics_df.columns]
                
                if available_score_cols:
                    gt_metrics_df['Average_Score'] = gt_metrics_df[available_score_cols].mean(axis=1)
                    ranking_df = gt_metrics_df[['Model', 'Average_Score'] + available_score_cols].sort_values('Average_Score', ascending=False)
                    
                    st.dataframe(ranking_df.style.background_gradient(subset=['Average_Score'], cmap='RdYlGn'))
                
                # Download ground truth metrics
                csv_buffer = io.StringIO()
                gt_metrics_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Ground Truth Performance Metrics",
                    data=csv_buffer.getvalue(),
                    file_name="ground_truth_performance_metrics.csv",
                    mime="text/csv"
                )

# Instructions
if train_file is None and not pretrained_models:
    st.info("""
    ## Instructions:
    
    ### Option 1: Train New Models
    1. Upload your training CSV file using the sidebar
    2. Adjust model parameters if needed
    3. Click 'Train Models' to train all algorithms
    4. View model performance and ROC curves
    5. Upload test data to make predictions
    6. Download trained models and predictions
    
    ### Option 2: Use Pre-trained Models
    1. Upload your pre-trained model files (.pkl) using the sidebar:
       - logistic_regression_model.pkl
       - random_forest_model.pkl  
       - decision_tree_model.pkl
       - kneighbours_model.pkl
    2. Upload test data to make predictions
    3. Select from available pre-trained models
    4. Download predictions
    
    ## Expected Data Format:
    - CSV files with columns: 'seq', 'Beta', 'Regulatory_Feature_Group', 'Relation_to_UCSC_CpG_Island'
    - 'seq' should contain DNA sequences
    - 'Beta' should contain binary methylation states (0/1)
    
    ## Pre-trained Models:
    Your well-trained models are ready to use! Simply upload the .pkl files to start making predictions immediately.
    """)