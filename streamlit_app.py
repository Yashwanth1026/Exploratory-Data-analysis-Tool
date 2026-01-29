import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import hashlib
import preprocessing_pipeline as pipeline
from EDA import (
    display_dataset_overview, display_missing_values, display_data_types,
    display_statistics_visualization, search_column, display_individual_feature_distribution,
    display_scatter_plot_of_two_numeric_features, categorical_variable_analysis,
    feature_exploration_numerical_variables, categorical_numerical_variable_analysis
)

st.set_page_config(page_title="Advanced Automated EDA & Preprocessing", layout="wide")
st.title("📊 Advanced Automated EDA & Data Preprocessing Tool")

# Initialize session state
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None

# Upload file
file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx", "xls"], key="data_uploader")

def load_data(file):
    """Loads data from the uploaded file."""
    try:
        if file.name.endswith('.csv'):
            encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding)
                    if not df.empty:
                        return df
                except UnicodeDecodeError:
                    continue
            st.error("Failed to decode CSV file. Please ensure it is valid with UTF-8 or Latin-1 encoding.")
            return None
            
        elif file.name.endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(file)
                if df.empty:
                    st.error("Uploaded file is empty")
                    return None
                return df
            except Exception as e:
                st.error(f"Error reading Excel file: {e}")
                return None
                
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None
    
    return None

def save_to_history(df):
    """Saves the current dataframe state to history."""
    if len(st.session_state.history) >= 10:
        st.session_state.history.pop(0)
    st.session_state.history.append(df.copy())

def undo_last_action():
    """Undoes the last preprocessing action."""
    if len(st.session_state.history) > 1:
        st.session_state.history.pop()
        return st.session_state.history[-1].copy()
    elif len(st.session_state.history) == 1:
        return st.session_state.history[-1].copy()
    return None

def validate_columns(df, columns, operation_name):
    """Validates if columns exist in the dataframe."""
    if not columns:
        st.warning(f"No columns selected for {operation_name}.")
        return False
    missing = [col for col in columns if col not in df.columns]
    if missing:
        st.error(f"Cannot {operation_name}: Columns not found - {', '.join(missing)}")
        return False
    return True

def show_data_quality_insights(df):
    """Displays data quality metrics."""
    st.subheader("🔍 Data Quality Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        type_issues = pipeline.data_type_conversion.detect_data_type_issues(df)
        if type_issues:
            st.warning("**Data Type Issues Found:**")
            for issue in type_issues[:5]:
                st.write(f"• {issue}")
            if len(type_issues) > 5:
                st.write(f"... and {len(type_issues) - 5} more issues.")
        else:
            st.success("✅ No data type issues detected")

    with col2:
        dup_info = pipeline.duplicates.detect_duplicates(df)
        st.metric("Duplicate Rows", dup_info['total_duplicates'],
                 f"{dup_info['duplicate_percentage']:.1f}%")

    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100 if not df.empty else 0
        st.metric("Missing Data", f"{missing_pct:.1f}%")

def show_advanced_preprocessing(df):
    """Displays advanced preprocessing options in the sidebar."""
    st.sidebar.header("🚀 Advanced Preprocessing")

    # Data Type Conversion
    with st.sidebar.expander("🔄 Data Type Conversion"):
        if st.checkbox("Convert Data Types", key="convert_types_check"):
            cols = st.multiselect("Select columns to convert", df.columns.tolist(), key="convert_types_cols")
            target_type = st.selectbox("Target type",
                                     ['datetime', 'numeric', 'category', 'string', 'boolean'],
                                     key="convert_types_target")
            if st.button("Convert Types", key="convert_types_btn"):
                if validate_columns(df, cols, "convert data types"):
                    df, _ = pipeline.data_type_conversion.convert_data_types(df, cols, target_type)
                    save_to_history(df)
                    st.session_state.processed_df = df
                    st.session_state.changed_columns = cols
                    st.rerun()

    # Datetime Features
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if datetime_cols:
        with st.sidebar.expander("📅 Datetime Features"):
            if st.checkbox("Extract Datetime Features", key="datetime_feature_check"):
                selected_col = st.selectbox("Select datetime column", datetime_cols, key="datetime_col_select")
                features = st.multiselect("Features to extract",
                                        ['year', 'month', 'day', 'weekday', 'hour',
                                         'minute', 'quarter', 'dayofyear'],
                                        default=['year', 'month', 'day'],
                                        key="datetime_features")
                if st.button("Extract Features", key="extract_datetime_btn"):
                    if validate_columns(df, [selected_col], "extract datetime features"):
                        df, _ = pipeline.datetime_features.datetime_feature_engineering(df, selected_col, features)
                        save_to_history(df)
                        st.session_state.processed_df = df
                        st.session_state.changed_columns = [selected_col]
                        st.rerun()

    # Text Preprocessing
    text_cols = [col for col in df.columns if df[col].dtype == 'object']
    if text_cols:
        with st.sidebar.expander("📝 Text Preprocessing"):
            if st.checkbox("Preprocess Text", key="text_preprocess_check"):
                selected_col = st.selectbox("Select text column", text_cols, key="text_col_select")
                steps = st.multiselect("Preprocessing steps",
                                     ['lowercase', 'remove_punct', 'remove_stopwords',
                                      'stemming', 'lemmatization', 'word_count',
                                      'char_count', 'remove_numbers', 'remove_whitespace'],
                                     default=['lowercase', 'remove_punct'],
                                     key="text_steps")
                if st.button("Process Text", key="process_text_btn"):
                    if validate_columns(df, [selected_col], "preprocess text"):
                        df, _ = pipeline.text_processing.text_preprocessing(df, selected_col, steps)
                        save_to_history(df)
                        st.session_state.processed_df = df
                        st.session_state.changed_columns = [selected_col]
                        st.rerun()

    # Advanced Imputation
    with st.sidebar.expander("🔧 Advanced Imputation"):
        imputation_method = st.selectbox("Imputation Method",
                                       ["KNN", "Interpolation", "Regression"],
                                       key="imputation_method")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if imputation_method == "KNN":
            knn_cols = st.multiselect("Select columns for KNN imputation", numeric_cols, key="knn_cols")
            n_neighbors = st.slider("Number of neighbors", 1, 20, 5, key="knn_neighbors")
            if st.button("Apply KNN Imputation", key="knn_impute_btn"):
                if validate_columns(df, knn_cols, "KNN imputation"):
                    df = pipeline.imputation.knn_imputation(df, knn_cols, n_neighbors)
                    save_to_history(df)
                    st.session_state.processed_df = df
                    st.session_state.changed_columns = knn_cols
                    st.rerun()

        elif imputation_method == "Interpolation":
            interp_cols = st.multiselect("Select columns for interpolation", numeric_cols, key="interp_cols")
            interp_method = st.selectbox("Interpolation method",
                                       ['linear', 'polynomial', 'spline'],
                                       key="interp_method")
            if st.button("Apply Interpolation", key="interpolate_btn"):
                if validate_columns(df, interp_cols, "interpolation"):
                    df = pipeline.imputation.interpolation_imputation(df, interp_cols, interp_method)
                    save_to_history(df)
                    st.session_state.processed_df = df
                    st.session_state.changed_columns = interp_cols
                    st.rerun()

        elif imputation_method == "Regression":
            target_col = st.selectbox("Target column for regression imputation", numeric_cols, key="reg_target_col")
            feature_cols = [col for col in numeric_cols if col != target_col]
            if st.button("Apply Regression Imputation", key="regression_impute_btn"):
                if validate_columns(df, [target_col] + feature_cols, "regression imputation"):
                    df, _ = pipeline.imputation.regression_imputation(df, numeric_cols, target_col)
                    save_to_history(df)
                    st.session_state.processed_df = df
                    st.session_state.changed_columns = [target_col] + numeric_cols
                    st.rerun()

    # Advanced Encoding
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_columns:
        with st.sidebar.expander("🎯 Advanced Encoding"):
            encoding_method = st.selectbox("Encoding Method",
                                         ["Target Encoding", "Frequency Encoding"],
                                         key="encoding_method")

            if encoding_method == "Target Encoding":
                target_col = st.selectbox("Target column", df.columns.tolist(), key="target_enc_target")
                cat_cols_for_encoding = st.multiselect("Categorical columns to encode", cat_columns, key="target_enc_cols")
                smoothing = st.slider("Smoothing factor", 0.1, 10.0, 1.0, key="smoothing_factor")
                if st.button("Apply Target Encoding", key="target_encode_btn"):
                    if validate_columns(df, cat_cols_for_encoding + [target_col], "target encoding"):
                        df = pipeline.encoding.target_encoding(df, cat_cols_for_encoding, target_col, smoothing)
                        save_to_history(df)
                        st.session_state.processed_df = df
                        st.session_state.changed_columns = cat_cols_for_encoding + [target_col]
                        st.rerun()

            elif encoding_method == "Frequency Encoding":
                freq_cols = st.multiselect("Select columns for frequency encoding", cat_columns, key="freq_enc_cols")
                if st.button("Apply Frequency Encoding", key="freq_encode_btn"):
                    if validate_columns(df, freq_cols, "frequency encoding"):
                        df = pipeline.encoding.frequency_encoding(df, freq_cols)
                        save_to_history(df)
                        st.session_state.processed_df = df
                        st.session_state.changed_columns = freq_cols
                        st.rerun()

    # Feature Selection
    with st.sidebar.expander("🎛️ Feature Selection"):
        selection_method = st.selectbox("Selection Method",
                                      ['variance', 'correlation', 'univariate'],
                                      key="selection_method")

        if selection_method == 'univariate':
            target_col = st.selectbox("Target Column", df.columns.tolist(), key="fs_target")
            k = st.slider("Number of features to keep", 1, len(df.columns)-1, min(10, len(df.columns)-1), key="k_features")
            if st.button("Apply Univariate Selection", key="univariate_select_btn"):
                if validate_columns(df, [target_col], "univariate selection"):
                    df, _ = pipeline.feature_selection.feature_selection(df, selection_method, target_col=target_col, k=k)
                    save_to_history(df)
                    st.session_state.processed_df = df
                    st.rerun()
        else:
            threshold = st.slider("Threshold", 0.0, 1.0, 0.8, key="threshold")
            if st.button(f"Apply {selection_method.title()} Selection", key=f"{selection_method}_select_btn"):
                df, _ = pipeline.feature_selection.feature_selection(df, selection_method, threshold=threshold)
                save_to_history(df)
                st.session_state.processed_df = df
                st.rerun()

    # Feature Transformation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        with st.sidebar.expander("🔄 Feature Transformation"):
            transform_method = st.selectbox("Transformation Method",
                                          ['log', 'sqrt', 'square', 'reciprocal',
                                           'boxcox', 'yeojohnson'],
                                          key="transform_method")
            transform_cols = st.multiselect("Select columns to transform", numeric_cols, key="transform_cols")
            if st.button("Apply Transformation", key="transform_features_btn"):
                if validate_columns(df, transform_cols, "feature transformation"):
                    df = pipeline.feature_transformation.transform_features(df, transform_cols, transform_method)
                    save_to_history(df)
                    st.session_state.processed_df = df
                    st.session_state.changed_columns = transform_cols
                    st.rerun()

        # Feature Binning
        with st.sidebar.expander("📊 Feature Binning"):
            bin_cols = st.multiselect("Select columns to bin", numeric_cols, key="bin_cols")
            bin_method = st.selectbox("Binning method", ['equal_width', 'equal_freq'], key="bin_method")
            n_bins = st.slider("Number of bins", 2, 20, 5, key="n_bins")
            if st.button("Apply Binning", key="bin_features_btn"):
                if validate_columns(df, bin_cols, "feature binning"):
                    df = pipeline.binning.bin_features(df, bin_cols, bin_method, n_bins)
                    save_to_history(df)
                    st.session_state.processed_df = df
                    st.session_state.changed_columns = bin_cols
                    st.rerun()

    # Principal Component Analysis
    if len(df.select_dtypes(include=[np.number]).columns) > 1:
        with st.sidebar.expander("📈 Principal Component Analysis"):
            pca_option = st.radio("PCA Option",
                                ["Fixed Components", "Variance Threshold"],
                                key="pca_option")

            if pca_option == "Fixed Components":
                max_components = len(df.select_dtypes(include=[np.number]).columns)
                n_components = st.number_input("Number of components",
                                             1, max_components, min(2, max_components),
                                             key="pca_n_components")
                if st.button("Apply PCA (Fixed)", key="pca_fixed_btn"):
                    df, pca_info = pipeline.pca_module.apply_pca(df, n_components=n_components)
                    save_to_history(df)
                    st.session_state.processed_df = df
                    st.session_state.pca_info = pca_info
                    st.session_state.changed_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    st.rerun()
            else:
                variance_threshold = st.slider("Variance to retain", 0.8, 0.99, 0.95, key="variance_threshold")
                if st.button("Apply PCA (Variance)", key="pca_variance_btn"):
                    df, pca_info = pipeline.pca_module.apply_pca(df, variance_threshold=variance_threshold)
                    save_to_history(df)
                    st.session_state.processed_df = df
                    st.session_state.pca_info = pca_info
                    st.session_state.changed_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    st.rerun()

    # Duplicate Handling
    with st.sidebar.expander("🔄 Duplicate Handling"):
        dup_info = pipeline.duplicates.detect_duplicates(df)
        st.write(f"Found {dup_info['total_duplicates']} duplicate rows")

        if dup_info['total_duplicates'] > 0:
            subset_cols = st.multiselect("Consider only these columns (optional)",
                                       df.columns.tolist(), key="dup_subset_cols")
            keep_option = st.selectbox("Keep which duplicate?", ['first', 'last', False], key="keep_option")
            if st.button("Remove Duplicates", key="remove_duplicates_btn"):
                df, _ = pipeline.duplicates.remove_duplicates(df, subset=subset_cols if subset_cols else None,
                                     keep=keep_option)
                save_to_history(df)
                st.session_state.processed_df = df
                st.session_state.changed_columns = []
                st.rerun()

def show_pca_visualization():
    """Displays PCA visualization if PCA was applied."""
    if 'pca_info' in st.session_state and st.session_state.pca_info is not None:
        st.subheader("📈 PCA Analysis Results")
        pca_info = st.session_state.pca_info

        col1, col2 = st.columns(2)

        with col1:
            if 'explained_variance_ratio' in pca_info:
                fig_var = px.bar(
                    x=range(1, len(pca_info['explained_variance_ratio']) + 1),
                    y=pca_info['explained_variance_ratio'],
                    title="Explained Variance by Component",
                    labels={'x': 'Component', 'y': 'Explained Variance Ratio'}
                )
                st.plotly_chart(fig_var, use_container_width=True)

        with col2:
            if 'cumulative_variance' in pca_info:
                fig_cum = px.line(
                    x=range(1, len(pca_info['cumulative_variance']) + 1),
                    y=pca_info['cumulative_variance'],
                    title="Cumulative Explained Variance",
                    labels={'x': 'Component', 'y': 'Cumulative Variance'}
                )
                if pca_info.get('variance_threshold') is not None:
                    fig_cum.add_hline(y=pca_info['variance_threshold'], 
                                    line_dash="dash", 
                                    line_color="red",
                                    annotation_text=f"{pca_info['variance_threshold']:.0%}")
                st.plotly_chart(fig_cum, use_container_width=True)

def main():
    """Main function to run the Streamlit app."""
    if file is not None:
        file_content = file.getvalue()
        file_content_hash = hashlib.md5(file_content).hexdigest()

        if st.session_state.file_hash != file_content_hash:
            st.session_state.file_hash = file_content_hash
            df = load_data(file)
            if df is None:
                st.session_state.original_df = None
                st.session_state.processed_df = None
                st.session_state.history = []
                st.session_state.file_hash = None
                if 'pca_info' in st.session_state:
                    del st.session_state.pca_info
                return

            st.session_state.original_df = df.copy()
            st.session_state.processed_df = df.copy()
            st.session_state.history = [df.copy()]
            if 'pca_info' in st.session_state:
                del st.session_state.pca_info
            st.rerun()

        if st.session_state.processed_df is None:
            st.warning("Data not loaded or failed to process.")
            return

        current_df = st.session_state.processed_df

        if st.sidebar.button("↩️ Undo Last Action", key="undo_btn"):
            undone_df = undo_last_action()
            if undone_df is not None:
                st.session_state.processed_df = undone_df
                if len(st.session_state.history) == 1 and 'pca_info' in st.session_state:
                    del st.session_state.pca_info
                st.rerun()
            else:
                st.sidebar.info("No more actions to undo.")

        if st.sidebar.button("🗑️ Reset to Original", key="reset_btn"):
            if st.session_state.original_df is not None:
                st.session_state.processed_df = st.session_state.original_df.copy()
                st.session_state.history = [st.session_state.original_df.copy()]
                if 'pca_info' in st.session_state:
                    del st.session_state.pca_info
                st.rerun()

        # Render advanced preprocessing options
        show_advanced_preprocessing(current_df)

        st.success(f"✅ Data Loaded Successfully! Shape: {current_df.shape}")
        num_columns, cat_columns = pipeline.utils.categorical_numerical(current_df)
        show_data_quality_insights(current_df)

        if st.session_state.original_df is not None and (len(current_df.columns) != len(st.session_state.original_df.columns) or current_df.shape != st.session_state.original_df.shape):
            st.info("🔄 **Data has been preprocessed:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Shape", f"{st.session_state.original_df.shape[0]} × {st.session_state.original_df.shape[1]}")
            with col2:
                st.metric("Current Shape", f"{current_df.shape[0]} × {current_df.shape[1]}")
            with col3:
                rows_change = current_df.shape[0] - st.session_state.original_df.shape[0]
                cols_change = current_df.shape[1] - st.session_state.original_df.shape[1]
                st.metric("Changes", f"Rows: {rows_change:+d}, Cols: {cols_change:+d}")

        tabs = st.tabs([
            "📋 Overview", "🔍 Missing Data", "🏷️ Data Types", "📊 Stats & Visuals",
            "🔎 Search", "📈 Feature Dist.", "🎯 Scatter Plot", "📊 Categorical Analysis",
            "🔬 Feature Exploration", "🔄 Cat vs Num", "📈 PCA Analysis", "🧪 Model Sandbox",
            "🆚 Compare Data"
        ])

        with tabs[0]:
            display_dataset_overview(current_df, cat_columns, num_columns)

        with tabs[1]:
            display_missing_values(current_df)

        with tabs[2]:
            display_data_types(current_df)

        with tabs[3]:
            display_statistics_visualization(current_df, cat_columns, num_columns)

        with tabs[4]:
            search_column(current_df)

        with tabs[5]:
            display_individual_feature_distribution(current_df, num_columns)

        with tabs[6]:
            display_scatter_plot_of_two_numeric_features(current_df, num_columns)

        with tabs[7]:
            categorical_variable_analysis(current_df, cat_columns)

        with tabs[8]:
            feature_exploration_numerical_variables(current_df, num_columns)

        with tabs[9]:
            categorical_numerical_variable_analysis(current_df, cat_columns, num_columns)

        with tabs[10]:
            show_pca_visualization()

        with tabs[11]:
            st.subheader("Model Sandbox")
            target_options = [c for c in current_df.columns]
            target_col = st.selectbox("Target Column", target_options, key="sandbox_target")
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, key="sandbox_test_size")
            if st.button("Train Models", key="sandbox_train_btn") and target_col:
                try:
                    results = pipeline.model_sandbox.train_and_evaluate(current_df, target_col, test_size=test_size)
                    st.write("Problem Type:", results.get("problem_type"))
                    metrics = results.get("metrics", {})
                    if metrics:
                        cols_metrics = st.columns(len(metrics))
                        i = 0
                        for k, v in metrics.items():
                            cols_metrics[i].metric(k.replace("_", " ").title(), f"{v:.4f}")
                            i += 1
                    fi = results.get("feature_importance")
                    if fi is not None and not fi.empty:
                        fig = px.bar(fi.head(20), x="feature", y="importance", title="Feature Importance (Top 20)")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error in model sandbox: {e}")

        with tabs[12]:
            st.subheader("🆚 Original vs. Processed Comparison")
            if st.session_state.original_df is not None:
                orig_df = st.session_state.original_df
                
                c1, c2 = st.columns(2)
                with c1:
                    st.info("Original Data")
                    st.write(f"**Shape:** {orig_df.shape}")
                    st.write(f"**Missing Values:** {orig_df.isnull().sum().sum()}")
                    st.write(f"**Duplicates:** {orig_df.duplicated().sum()}")
                with c2:
                    st.success("Processed Data")
                    st.write(f"**Shape:** {current_df.shape}")
                    st.write(f"**Missing Values:** {current_df.isnull().sum().sum()}")
                    st.write(f"**Duplicates:** {current_df.duplicated().sum()}")

                st.markdown("---")
                st.write("### 📊 Distribution Comparison")
                
                # Column comparison
                common_cols = sorted(list(set(orig_df.columns) & set(current_df.columns)))
                if common_cols:
                    comp_col = st.selectbox("Select Column to Compare", common_cols, key="compare_col_select")
                    
                    if comp_col in orig_df.columns and comp_col in current_df.columns:
                        is_num_orig = pd.api.types.is_numeric_dtype(orig_df[comp_col])
                        is_num_curr = pd.api.types.is_numeric_dtype(current_df[comp_col])
                        
                        if is_num_orig and is_num_curr:
                            import plotly.graph_objects as go
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(x=orig_df[comp_col], name='Original', opacity=0.75, marker_color='blue'))
                            fig.add_trace(go.Histogram(x=current_df[comp_col], name='Processed', opacity=0.75, marker_color='green'))
                            fig.update_layout(barmode='overlay', title=f"Distribution Comparison: {comp_col}", 
                                            xaxis_title=comp_col, yaxis_title="Count")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Stats comparison table
                            st.write("**Statistics Comparison**")
                            desc_orig = orig_df[comp_col].describe()
                            desc_curr = current_df[comp_col].describe()
                            comp_df = pd.DataFrame({'Original': desc_orig, 'Processed': desc_curr})
                            st.dataframe(comp_df)
                            
                        else:
                            st.info(f"Selected column '{comp_col}' is not numeric or has incompatible types for histogram comparison.")
                            st.write(f"Original Type: {orig_df[comp_col].dtype}, Processed Type: {current_df[comp_col].dtype}")
                else:
                    st.warning("No common columns found between original and processed data.")
            else:
                st.warning("Original data not available.")

        st.sidebar.header("🧹 Basic Preprocessing")

        if st.sidebar.checkbox("Remove Columns", key="remove_cols_check"):
            cols = st.sidebar.multiselect("Columns to Remove", current_df.columns.tolist(), key="remove_cols_select")
            if cols and st.sidebar.button("Remove Selected Columns", key="remove_cols_btn"):
                if validate_columns(current_df, cols, "remove columns"):
                    current_df = pipeline.utils.remove_selected_columns(current_df, cols)
                    save_to_history(current_df)
                    st.session_state.processed_df = current_df
                    st.rerun()

        if st.sidebar.checkbox("Remove Rows with NA", key="remove_na_check"):
            cols = st.sidebar.multiselect("Select Columns for NA Removal",
                                        current_df.columns.tolist(), key="remove_na_cols")
            if cols and st.sidebar.button("Remove NA Rows", key="remove_na_btn"):
                if validate_columns(current_df, cols, "remove NA rows"):
                    current_df = pipeline.missing_values.remove_rows_with_missing_data(current_df, cols)
                    save_to_history(current_df)
                    st.session_state.processed_df = current_df
                    st.rerun()

        if st.sidebar.checkbox("Fill Missing Values", key="fill_missing_check"):
            cols = st.sidebar.multiselect("Select Columns to Fill",
                                        current_df.columns.tolist(), key="fill_missing_cols")
            method = st.sidebar.radio("Method", ["mean", "median", "mode"], key="fill_method")
            if cols and st.sidebar.button("Fill Missing", key="fill_missing_btn"):
                if validate_columns(current_df, cols, "fill missing values"):
                    current_df = pipeline.missing_values.fill_missing_data(current_df, cols, method)
                    save_to_history(current_df)
                    st.session_state.processed_df = current_df
                    st.rerun()

        if st.sidebar.checkbox("Label Encoding", key="label_enc_check"):
            _, cat_columns_current = pipeline.utils.categorical_numerical(current_df)
            cols = st.sidebar.multiselect("Categorical Columns for Label Encoding", cat_columns_current, key="label_enc_cols")
            if cols and st.sidebar.button("Apply Label Encoding", key="label_enc_btn"):
                if validate_columns(current_df, cols, "label encoding"):
                    current_df = pipeline.encoding.label_encode(current_df, cols)
                    save_to_history(current_df)
                    st.session_state.processed_df = current_df
                    st.rerun()

        if st.sidebar.checkbox("One-Hot Encoding", key="onehot_check"):
            _, cat_columns_current = pipeline.utils.categorical_numerical(current_df)
            cols = st.sidebar.multiselect("Categorical Columns for One-Hot", cat_columns_current, key="onehot_cols")
            if cols and st.sidebar.button("Apply One-Hot Encoding", key="onehot_btn"):
                if validate_columns(current_df, cols, "one-hot encoding"):
                    current_df = pipeline.encoding.one_hot_encode(current_df, cols)
                    save_to_history(current_df)
                    st.session_state.processed_df = current_df
                    st.rerun()

        if st.sidebar.checkbox("Standard Scaling", key="std_scale_check"):
            num_columns_current, _ = pipeline.utils.categorical_numerical(current_df)
            cols = st.sidebar.multiselect("Numerical Columns to Scale (Standard)", num_columns_current, key="std_scale_cols")
            if cols and st.sidebar.button("Apply Standard Scaling", key="std_scale_btn"):
                if validate_columns(current_df, cols, "standard scaling"):
                    current_df = pipeline.scaling.standard_scale(current_df, cols)
                    save_to_history(current_df)
                    st.session_state.processed_df = current_df
                    st.rerun()

        st.sidebar.header("✂️ Row Management")

        with st.sidebar.expander("🔍 Global Search"):
            search_query = st.text_input("Search term (all columns)", key="global_search_input")
            case_sensitive = st.checkbox("Case sensitive", value=False, key="global_search_case")
            
            if st.button("Search & Filter", key="global_search_btn"):
                if search_query:
                    try:
                        # Use the modular filtering function
                        current_df = pipeline.filtering.search_rows_global(current_df, search_query, case=case_sensitive)
                        
                        st.success(f"Found {len(current_df)} rows matching '{search_query}'.")
                        save_to_history(current_df)
                        st.session_state.processed_df = current_df
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error searching: {e}")
                else:
                    st.warning("Please enter a search term.")

        with st.sidebar.expander("📊 Filter Rows"):
            if st.checkbox("Apply Row Filter", key="filter_rows_check"):
                filter_col = st.selectbox("Select column to filter", current_df.columns.tolist(), key="filter_col_select")
                if filter_col:
                    col_dtype = current_df[filter_col].dtype
                    
                    if pd.api.types.is_numeric_dtype(col_dtype):
                        operators = ['>', '<', '==', '!=', '>=', '<=']
                        filter_value = st.number_input(f"Enter value for {filter_col}", key="filter_value_input_num")
                    elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                        operators = ['>', '<', '==', '!=', '>=', '<=']
                        filter_value = st.date_input(f"Enter date for {filter_col}", key="filter_value_input_date")
                    else:
                        operators = ['==', '!=', 'contains', 'starts with', 'ends with']
                        filter_value = st.text_input(f"Enter value for {filter_col}", key="filter_value_input_text")

                    filter_operator = st.selectbox("Select operator", operators, key="filter_operator_select")

                    if st.button("Apply Filter", key="apply_filter_btn"):
                        try:
                            # Use the modular filtering function
                            current_df = pipeline.filtering.filter_dataframe(current_df, filter_col, filter_value, filter_operator)
                            
                            st.success(f"Filtered data: {len(st.session_state.processed_df) - len(current_df)} rows removed.")
                            save_to_history(current_df)
                            st.session_state.processed_df = current_df
                            st.session_state.changed_columns = []
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error applying filter: {e}")

        with st.sidebar.expander("🗑️ Drop Rows"):
            if st.checkbox("Apply Drop Rows", key="drop_rows_check"):
                drop_col = st.selectbox("Select column to drop rows based on", current_df.columns.tolist(), key="drop_col_select")
                if drop_col:
                    col_dtype = current_df[drop_col].dtype

                    if pd.api.types.is_numeric_dtype(col_dtype):
                        operators = ['>', '<', '==', '!=', '>=', '<=']
                        drop_value = st.number_input(f"Enter value for {drop_col}", key="drop_value_input_num")
                    elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                        operators = ['>', '<', '==', '!=', '>=', '<=']
                        drop_value = st.date_input(f"Enter date for {drop_col}", key="drop_value_input_date")
                    else:
                        operators = ['==', '!=', 'contains', 'starts with', 'ends with']
                        drop_value = st.text_input(f"Enter value for {drop_col}", key="drop_value_input_text")

                    drop_operator = st.selectbox("Select operator", operators, key="drop_operator_select")

                    if st.button("Apply Drop", key="apply_drop_btn"):
                        try:
                            # Use the modular filtering function
                            current_df = pipeline.filtering.drop_rows_based_on_filter(current_df, drop_col, drop_value, drop_operator)
                            
                            st.success(f"Dropped data: {len(st.session_state.processed_df) - len(current_df)} rows removed.")
                            save_to_history(current_df)
                            st.session_state.processed_df = current_df
                            st.session_state.changed_columns = []
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error applying drop: {e}")

        st.sidebar.header("🧠 Recommendations")
        try:
            recs = pipeline.recommendations.generate_recommendations(current_df)
            if recs:
                st.sidebar.write(f"Found {len(recs)} suggestions:")
                for r in recs:
                    # Determine icon based on severity
                    sev = r.get('severity', 'Low')
                    icon = "🔴" if sev == "High" else "🟠" if sev == "Medium" else "🟡"
                    
                    # Create an expander for each recommendation
                    with st.sidebar.expander(f"{icon} {r['category']}: {r['column']}"):
                        st.markdown(f"**Issue:** {r['message']}")
                        st.markdown(f"**Why it matters:**\n{r['explanation']}")
                        st.info(f"💡 **Suggestion:** {r['suggestion']}")
            else:
                st.sidebar.success("✅ No major issues found. Data looks good!")
        except Exception as e:
            st.sidebar.error(f"Recommendation error: {e}")

        st.sidebar.header("⬇️ Export")
        csv_bytes = pipeline.code_export.dataframe_to_csv_bytes(current_df)
        st.sidebar.download_button("Download Cleaned CSV", data=csv_bytes, file_name="cleaned_dataset.csv", mime="text/csv", key="download_csv_btn")
        
        # Summary Report
        report_text = f"Dataset Report\nGenerated on {pd.Timestamp.now()}\n\n"
        report_text += f"Rows: {current_df.shape[0]}\nColumns: {current_df.shape[1]}\n\n"
        report_text += "Column Info:\n" + str(current_df.dtypes) + "\n\n"
        report_text += "Missing Values:\n" + str(current_df.isnull().sum()) + "\n\n"
        report_text += "Summary Statistics:\n" + str(current_df.describe())
        st.sidebar.download_button("Download Summary Report", data=report_text, file_name="data_summary.txt", mime="text/plain", key="download_report_btn")

        try:
            pipeline_code = pipeline.code_export.generate_sklearn_pipeline_code(current_df)
            st.sidebar.download_button("Download Preprocessing Pipeline Code", data=pipeline_code, file_name="preprocessing_pipeline.py", mime="text/plain", key="download_code_btn")
        except Exception as e:
            st.sidebar.error(f"Code generation failed: {e}")
    else:
        st.info("Upload a CSV or Excel file to get started.")

if __name__ == "__main__":
    main()
