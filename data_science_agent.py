import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
from tabulate import tabulate
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
import io
import os

# --- TOOLS for Chat Agent ---
# --- TOOLS for Chat Agent ---
@tool
def get_dataset_info(ignore: str = None) -> str:
    """Returns basic information about the loaded dataset (columns, dtypes, head)."""
    if "df" not in st.session_state or st.session_state.df is None:
        return "No data loaded."
    
    df = st.session_state.df
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    head_str = df.head().to_markdown()
    return f"Dataset Info:\n{info_str}\n\nFirst 5 rows:\n{head_str}"

@tool
def get_summary_statistics(ignore: str = None) -> str:
    """Calculates descriptive statistics for numeric columns."""
    if "df" not in st.session_state or st.session_state.df is None:
        return "No data loaded."
    return st.session_state.df.describe().to_markdown()

@tool
def run_linear_regression(target_col: str = None, feature_cols: list[str] = None, ignore: str = None) -> str:
    """Runs a Linear Regression model. Args: target_col (string), feature_cols (list of strings)."""
    if "df" not in st.session_state or st.session_state.df is None: return "No data."
    
    # Handle JSON string passed as first arg
    import json
    if target_col and isinstance(target_col, str) and target_col.strip().startswith("{"):
        try:
            params = json.loads(target_col.replace("'", '"')) # Simple fix for single quotes
            target_col = params.get("target_col")
            feature_cols = params.get("feature_cols")
        except:
            pass

    if not target_col or not feature_cols:
        return "Error: target_col and feature_cols are required."

    df = st.session_state.df
    try:
        model_df = df[[target_col] + feature_cols].dropna()
        X = model_df[feature_cols]
        y = model_df[target_col]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return str(model.summary())
    except Exception as e:
        return f"Regression failed: {e}"

@tool
def run_svm_regression(target_col: str = None, feature_col: str = None, ignore: str = None) -> str:
    """Runs Support Vector Regression (SVR) for a single feature. Args: target_col, feature_col."""
    if "df" not in st.session_state or st.session_state.df is None: return "No data."
    
    # Handle JSON string passed as first arg
    import json
    if target_col and isinstance(target_col, str) and target_col.strip().startswith("{"):
        try:
            params = json.loads(target_col.replace("'", '"'))
            target_col = params.get("target_col")
            feature_col = params.get("feature_col")
        except:
            pass

    if not target_col or not feature_col:
        return "Error: target_col and feature_col are required."

    df = st.session_state.df
    try:
        model_df = df[[target_col, feature_col]].dropna()
        X = model_df[[feature_col]]
        y = model_df[target_col]
        svr = SVR()
        svr.fit(X, y)
        y_pred = svr.predict(X)
        r2 = r2_score(y, y_pred)
        return f"SVR Trained. R2 Score: {r2:.4f}"
    except Exception as e:
        return f"SVM failed: {e}"

@tool
def run_manova(dependent_vars: list[str] = None, independent_var: str = None, ignore: str = None) -> str:
    """Runs MANOVA. Args: dependent_vars (list of cols), independent_var (categorical group col)."""
    if "df" not in st.session_state or st.session_state.df is None: return "No data."
    
    # Handle JSON string passed as first arg - simpler heuristic for list
    # ReAct agent might pass the whole input string differently for lists
    # Note: dependent_vars is list[str], so strict typing might still fail if string passed. 
    # But let's try to be robust. 
    
    if not dependent_vars or not independent_var:
        return "Error: dependent_vars and independent_var are required."

    df = st.session_state.df
    try:
        deps_str = " + ".join(dependent_vars)
        formula = f"{deps_str} ~ {independent_var}"
        manova = MANOVA.from_formula(formula, data=df)
        return str(manova.mv_test())
    except Exception as e:
        return f"MANOVA failed: {e}"

@tool
def generate_plot(plot_type: str, x_col: str = None, y_col: str = None, hue: str = None, ignore: str = None) -> str:
    """Generates a plot. REQUIRED: plot_type (one of 'scatter', 'bar', 'box', 'histogram', 'heatmap'), x_col (column name). OPTIONAL: y_col, hue."""
    if "df" not in st.session_state or st.session_state.df is None: return "No data."
    if not x_col and plot_type != 'heatmap': return "Error: x_col argument is required for this plot type."
    
    df = st.session_state.df
    try:
        fig, ax = plt.subplots()
        if plot_type == 'scatter':
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
        elif plot_type == 'bar':
            sns.barplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
        elif plot_type == 'box':
            sns.boxplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax)
        elif plot_type == 'histogram':
            sns.histplot(data=df, x=x_col, kde=True, hue=hue, ax=ax)
        elif plot_type == 'heatmap':
            numeric_df = df.select_dtypes(include=[np.number])
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        
        if "plots" not in st.session_state:
            st.session_state.plots = []
        st.session_state.plots.append(fig)
        return f"Plot ({plot_type}) generated and saved to display queue."
    except Exception as e:
        return f"Plotting failed: {e}"

@tool
def run_automl(target_col: str, task_type: str = 'classification', ignore: str = None) -> str:
    """
    Runs AutoML to find the best model for a given target column.
    Args:
        target_col (str): The name of the target column.
        task_type (str): 'classification' or 'regression'.
    Returns:
        str: A table summarizing the performance of different models and the best model's parameters.
    """
    if "df" not in st.session_state or st.session_state.df is None: return "No data."
    df = st.session_state.df.copy() # Work on a copy
    
    if target_col not in df.columns:
        return f"Error: Target column '{target_col}' not found."
    
    try:
        # 1. Preprocessing
        # Drop rows with missing target
        df = df.dropna(subset=[target_col])
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Identify columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
        # 2. Model Selection & Grids
        models = {}
        if task_type.lower() == 'classification':
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            
            # Encode target if classification
            le = LabelEncoder()
            y = le.fit_transform(y)
            
            models = {
                'LogisticRegression': (LogisticRegression(max_iter=1000), {'classifier__C': [0.1, 1.0, 10.0]}),
                'RandomForest': (RandomForestClassifier(), {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10, 20]}),
                'SVM': (SVC(), {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}),
                'XGBoost': (xgb.XGBClassifier(eval_metric='logloss'), {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1]}),
                'LightGBM': (lgb.LGBMClassifier(verbose=-1), {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1]})
            }
            metric = 'accuracy'
        else:
            from sklearn.linear_model import LinearRegression
            from sklearn.svm import SVR
            
            models = {
                'LinearRegression': (LinearRegression(), {}),
                'RandomForest': (RandomForestRegressor(), {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10, 20]}),
                'SVM': (SVR(), {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}),
                'XGBoost': (xgb.XGBRegressor(), {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1]}),
                'LightGBM': (lgb.LGBMRegressor(verbose=-1), {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1]})
            }
            metric = 'r2'

        # 3. Training & Evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = []
        best_score = -np.inf
        best_model_name = ""
        
        for name, (model, params) in models.items():
            # Create Pipeline
            # Note: 'classifier' prefix in params matches this step name
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model)])
            
            # RandomizedSearchCV
            # For speed, n_iter is small
            search = RandomizedSearchCV(clf, params, n_iter=5, cv=3, scoring=metric, n_jobs=-1, random_state=42)
            search.fit(X_train, y_train)
            
            score = search.score(X_test, y_test)
            results.append([name, f"{score:.4f}", str(search.best_params_)])
            
            if score > best_score:
                best_score = score
                best_model_name = name

        # Format Output
        headers = ["Model", f"Test Score ({metric})", "Best Params"]
        table = tabulate(results, headers=headers, tablefmt="github")
        
        tested_models_str = ", ".join(models.keys())
        
        return f"### AutoML Results for Target: {target_col}\n\nTask: {task_type}\nWinner: **{best_model_name}** ({best_score:.4f})\n\n**Tested Models:** {tested_models_str}\n\n{table}"
        
    except Exception as e:
        return f"AutoML failed: {e}"


def main():
    st.set_page_config(page_title="Data Science Agent (NVIDIA Powered)", page_icon="🧪", layout="wide")
    st.title("🧪 Data Science Agent")
    st.caption("Powered by NVIDIA NIM & LangChain")

    # Initialize Session State
    if "df" not in st.session_state:
        st.session_state.df = None
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI Data Scientist. Upload a dataset, and I'll generate a summary for you."}]
    if "summary_generated" not in st.session_state:
        st.session_state.summary_generated = False
    
    # --- Sidebar for Config & Upload ---
    with st.sidebar:
        st.header("1. Configuration")
        # Default user key
        default_key = "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"
        api_key = st.text_input("API Key (NVIDIA or OpenRouter)", value=default_key, type="password")
        
        st.header("2. Data Upload")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        
        if uploaded_file and (st.session_state.df is None or uploaded_file.name != getattr(st.session_state, 'file_name', '')):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                st.session_state.file_name = uploaded_file.name
                st.session_state.summary_generated = False # Reset summary trigger on new file
                st.success(f"Loaded {len(df)} rows.")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    # --- Unified Layout ---
    
    # 1. Manual Workflow (Collapsible)
    with st.expander("🛠️ Manual Step-by-Step Analysis Tools", expanded=False):
        if st.session_state.df is not None:
            df = st.session_state.df
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("1. Data Overview")
                st.dataframe(df.head())
                st.caption(f"Shape: {df.shape}")
            
            with col2:
                st.subheader("2. Statistics")
                st.text(tabulate(df.describe(), headers='keys', tablefmt='psql'))
                
            st.divider()
            
            # Need to import these inside main or at top level if mostly used here
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            
            st.divider()
            
            col3, col4 = st.columns(2)
            
            # Viz
            with col3:
                st.subheader("3. Visualization")
                viz_type = st.selectbox("Choose Visualization", ["Correlation Heatmap", "Pairplot", "Distribution"], key="manual_viz")
                if viz_type == "Correlation Heatmap":
                    numeric_df = df.select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        fig, ax = plt.subplots()
                        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
                        st.pyplot(fig)
                elif viz_type == "Distribution":
                    target_col = st.selectbox("Column", df.columns, key="manual_dist_col")
                    fig, ax = plt.subplots()
                    sns.histplot(df[target_col], kde=True, ax=ax)
                    st.pyplot(fig)

            # Modeling
            with col4:
                st.subheader("4. Quick Modeling & AutoML")
                target = st.selectbox("Target Variable", df.columns, key="manual_target")
                model_choice = st.selectbox("Choose Model", ["Random Forest", "XGBoost", "LightGBM", "SVM", "Linear/Logistic Regression"], key="model_choice")
                
                if st.button("Train Model", key="manual_train"):
                    try:
                        # Simple Prep
                        model_df = df.dropna().copy()
                        for col in model_df.select_dtypes(include=['object']).columns:
                            model_df[col] = model_df[col].astype('category').cat.codes
                        
                        X = model_df.drop(columns=[target])
                        y = model_df[target]
                        
                        task = "Regression" if len(y.unique()) > 20 and y.dtype != 'object' else "Classification"
                        
                        if model_choice == "Random Forest":
                            model = RandomForestRegressor() if task == "Regression" else RandomForestClassifier()
                        elif model_choice == "XGBoost":
                            model = xgb.XGBRegressor() if task == "Regression" else xgb.XGBClassifier()
                        elif model_choice == "LightGBM":
                            model = lgb.LGBMRegressor(verbose=-1) if task == "Regression" else lgb.LGBMClassifier(verbose=-1)
                        elif model_choice == "SVM":
                            model = SVR() if task == "Regression" else SVC()
                        else:
                            model = LinearRegression() if task == "Regression" else LogisticRegression(max_iter=1000)
                            
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model.fit(X_train, y_train)
                        score = model.score(X_test, y_test)
                        
                        metric_name = "R2 Score" if task == "Regression" else "Accuracy"
                        st.success(f"**{model_choice} ({task})** | **{metric_name}:** {score:.4f}")
                        
                        if task == "Regression":
                            fig, ax = plt.subplots()
                            sns.scatterplot(x=y_test, y=model.predict(X_test), ax=ax)
                            plot_max = max(y_test.max(), model.predict(X_test).max())
                            plot_min = min(y_test.min(), model.predict(X_test).min())
                            plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--')
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Modeling Error: {e}")

                st.divider()
                if st.button("🚀 Run AutoML (Compare All)", key="btn_automl"):
                    with st.spinner("Running AutoML..."):
                         # Detect task
                         temp_df = df.dropna(subset=[target])
                         u_len = len(temp_df[target].unique())
                         detected_task = "regression" if u_len > 20 and temp_df[target].dtype != 'object' else "classification"
                         
                         result = run_automl.invoke({"target_col": target, "task_type": detected_task})
                         st.markdown(result)
        else:
            st.info("Upload data in the sidebar to enable manual tools.")

    st.divider()

    # 2. Conversational AI
    st.subheader("💬 AI Data Assistant")

    # --- Auto-Summary Logic ---
    if st.session_state.df is not None and not st.session_state.summary_generated:
        with st.status("Analyzing Data & Generating Executive Summary...", expanded=True) as status:
            try:
                df = st.session_state.df
                # Prepare info for LLM
                buffer = io.StringIO()
                df.info(buf=buffer)
                info_text = buffer.getvalue()
                desc_text = df.describe().to_markdown()
                head_text = df.head().to_markdown()
                
                context = f"Dataset Info:\n{info_text}\n\nStats:\n{desc_text}\n\nFirst Rows:\n{head_text}"
                
                # Call NVIDIA LLM
                # Init LLM based on Key Type
                if api_key.startswith("sk-or-"):
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        model="meta-llama/llama-3.1-70b-instruct", 
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1"
                    )
                else:
                    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=api_key)
                prompt = f"You are a Data Scientist. Analyze the following dataset summary and provide a comprehensive executive summary of the data, highlighting key distributions, potential data quality issues, and interesting patterns. Use bullet points.\n\n{context}"
                
                response = llm.invoke(prompt)
                summary_text = response.content
                
                # Add to chat history
                display_msg = f"**Data Executive Summary**\n\n{summary_text}\n\n*You can now ask me questions about this data!*"
                st.session_state.messages.append({"role": "assistant", "content": display_msg})
                st.session_state.summary_generated = True
                status.update(label="Summary Generated!", state="complete", expanded=False)
                st.rerun() # Refresh to show message
                
            except Exception as e:
                status.update(label="Summary Generation Failed", state="error")
                st.error(f"Error generating summary: {e}")

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Display Queued Plots
    if "plots" in st.session_state and st.session_state.plots:
        with st.chat_message("assistant"):
            for fig in st.session_state.plots:
                st.pyplot(fig)
        st.session_state.plots = [] # Clear

    # Chat Input
    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.df is not None:
            with st.chat_message("assistant"):
                with st.spinner("Thinking (NVIDIA NIM)..."):
                        try:
                            # Initialize NVIDIA Chat Model
                            # Initialize Chat Model
                            if api_key.startswith("sk-or-"):
                                from langchain_openai import ChatOpenAI
                                llm = ChatOpenAI(
                                    model="meta-llama/llama-3.1-70b-instruct", 
                                    api_key=api_key,
                                    base_url="https://openrouter.ai/api/v1"
                                )
                            else:
                                llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=api_key)
                            
                            tools = [get_dataset_info, get_summary_statistics, run_linear_regression, run_svm_regression, run_manova, generate_plot, run_automl]
                            
                            # Custom ReAct Prompt
                            template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''
                            
                            from langchain.agents import create_react_agent
                            from langchain_core.prompts import PromptTemplate
                            
                            prompt_template = PromptTemplate.from_template(template)
                            agent = create_react_agent(llm, tools, prompt_template)
                            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
                            
                            response = agent_executor.invoke({"input": prompt}) # Removed chat_history for ReAct simplicity or append to input
                            output_text = response['output']
                            
                            st.markdown(output_text)
                            st.session_state.messages.append({"role": "assistant", "content": output_text})
                            
                            # Show plots if any were generated by tools
                            if "plots" in st.session_state and st.session_state.plots:
                                for fig in st.session_state.plots:
                                    st.pyplot(fig)
                                st.session_state.plots = []
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            with st.chat_message("assistant"):
                st.warning("Please upload a dataset in the sidebar first.")

if __name__ == "__main__":
    main()
