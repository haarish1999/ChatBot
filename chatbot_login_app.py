import json
import streamlit as st
import sqlite3
import bcrypt
import time
import pandas as pd 
from io import BytesIO
from sqlite3 import Binary 
import random 
import numpy as np
import datetime

try:
    import spacy
except ImportError:
    spacy = None

# ==============================
# DATA UTILITY FUNCTIONS
# ==============================

def json_to_simple_entities(entities_json):
    """Converts a JSON string of entities (from DB) back to the simple k:v, k:v string for the UI."""
    if not entities_json: 
        return ""
    try:
        entities_dict = json.loads(entities_json)
        return ", ".join(f"{k}:{v}" for k, v in entities_dict.items())
    except (json.JSONDecodeError, TypeError):
        return ""
    
def get_db_connection():
    # Placeholder: Assuming this connects to your users.db
    return sqlite3.connect("users.db", check_same_thread=False)

def get_system_metrics():
    """Retrieves key counts from core tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        def fetch_count(table_name):
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]

        metrics = {
            'total_users': fetch_count("users"),
            'total_workspaces': fetch_count("workspaces"),
            'total_annotations': fetch_count("annotations"),
            'total_models': fetch_count("models")
        }
        return metrics
    except sqlite3.OperationalError as e:
        st.error(f"DB Error fetching metrics: {e}. Ensure all tables (users, workspaces, annotations, models) exist.")
        return {'total_users': 0, 'total_workspaces': 0, 'total_annotations': 0, 'total_models': 0}
    finally:
        conn.close()

def get_db_connection_local():
    """Creates a fresh, local connection for temporary operations."""
    # Use a new connection object, not the global one.
    return sqlite3.connect("users.db", check_same_thread=False)

def get_all_management_data():
    """Retrieves comprehensive data for all management tables using a local, fresh connection."""
    
    # CRITICAL FIX: Use the local connection function explicitly
    conn_local = get_db_connection_local()
    
    # Default empty DataFrames in case of failure
    user_df = pd.DataFrame()
    workspace_df = pd.DataFrame()
    model_df = pd.DataFrame()
    
    try:
        # User data: Name, Email
        # NOTE: Using conn_local for all read_sql calls
        # CRITICAL: Ensure the columns are retrieved correctly
        user_df = pd.read_sql("SELECT name, email FROM users", conn_local) 
        
        # Workspace data: Linked to user
        workspace_df = pd.read_sql("SELECT user_email, workspace_name, domain, last_modified FROM workspaces", conn_local)
        
        # Model data: Evaluation history
        model_df = pd.read_sql("SELECT user_email, workspace_name, version, accuracy, f1_score, train_date, notes, trained_samples FROM models", conn_local)

        # Fetch Annotations data (includes NLU corrections and general feedback)
        annotations_df = pd.read_sql("SELECT user_email, workspace_name, sentence, intent, entities_json, remarks, last_modified FROM annotations ORDER BY last_modified DESC", get_db_connection_local())
        
        return user_df, workspace_df, model_df
    
    except Exception as e:
        # The error is caught here. Return empty DFs, but ensure the error handling in show_admin_portal
        # can manage it (which is what we will fix in the next block).
        st.error(f"Error fetching admin data: {e}. Check if tables exist.")
        return user_df, workspace_df, model_df
        
    finally:
        # CRITICAL FIX: Close the local connection to prevent conflicts
        conn_local.close()

# Keep the global definition of get_db_connection() for the cached global 'conn' object

# ==============================
# DATABASE SETUP (THREAD-SAFE CONNECTION & MIGRATION)
# ==============================
@st.cache_resource
def get_db_connection():
    """Creates and caches the SQLite connection object (Do NOT close this connection)."""
    conn = sqlite3.connect("users.db", check_same_thread=False)
    # Set to IMMEDIATE to handle schema creation robustly
    conn.isolation_level = None 
    return conn
# Correction for delete_model_version:

def delete_model_version(version_name):
    """Executes the SQL DELETE query to remove a model version using the stable context manager."""
    
    try:
        with sqlite3.connect("users.db", check_same_thread=False) as conn_local:
            cursor = conn_local.cursor()
            
            cursor.execute(
                "DELETE FROM models WHERE version = ?",
                (version_name,)
            )
            # Commit and close handled by 'with'
            
            st.success(f"✅ Successfully deleted model version: {version_name}")
            
    except Exception as e:
        st.error(f"Failed to delete model version {version_name}: {e}")
        return
        
    # CRITICAL FIX: Clear BOTH caches and rerun to refresh the page display and dropdown
    try:
        # Clear st.cache_resource (for DB connection)
        st.cache_resource.clear() 
        # Clear st.cache_data (for data loading functions used elsewhere, for robustness)
        st.cache_data.clear() 
    except Exception:
        pass
    
    # Must rerun to display the success message and reload the admin page data
    st.rerun()
# Correction for delete_user_data:

def delete_user_data(user_email_to_delete):
    """
    Deletes the user and ALL associated data (cascading delete) using the
    stable SQLite context manager pattern.
    """
    
    if not user_email_to_delete or user_email_to_delete.strip() == "":
        st.error("Please provide a valid email to delete.")
        return

    # Use the SQLite context manager (with sqlite3.connect) for stability
    try:
        with sqlite3.connect("users.db", check_same_thread=False) as conn_local:
            cursor = conn_local.cursor()
            
            # --- EXECUTE ALL DELETE QUERIES (Cascading Delete) ---
            cursor.execute("DELETE FROM models WHERE user_email = ?", (user_email_to_delete,))
            cursor.execute("DELETE FROM annotations WHERE user_email = ?", (user_email_to_delete,))
            cursor.execute("DELETE FROM datasets WHERE user_email = ?", (user_email_to_delete,))
            cursor.execute("DELETE FROM workspaces WHERE user_email = ?", (user_email_to_delete,))
            cursor.execute("DELETE FROM users WHERE email = ?", (user_email_to_delete,))
            
            # The 'with' statement handles conn_local.commit() on success and conn_local.close() automatically.
            
            st.success(f"✅ Successfully deleted user and all associated data for: {user_email_to_delete}")
            
    except Exception as e:
        st.error(f"Failed to delete user {user_email_to_delete}: Database Error: {e}")
        return
        
    # CRITICAL FIX: Clear cache and rerun to refresh the page display and user list
    try:
        st.cache_resource.clear()
        # The global 'conn' will be re-created on the next run
    except Exception:
        pass

    st.rerun()
def delete_workspace_data(user_email_to_delete, workspace_name_to_delete):
    """
    Deletes the specific workspace and ALL associated data (cascading delete).
    """
    
    if not workspace_name_to_delete or workspace_name_to_delete.strip() == "":
        st.error("Please provide a valid workspace name to delete.")
        return

    try:
        with sqlite3.connect("users.db", check_same_thread=False) as conn_local:
            cursor = conn_local.cursor()
            
            # --- EXECUTE ALL DELETE QUERIES (Cascading Delete) ---
            # Delete models associated with the workspace
            cursor.execute("DELETE FROM models WHERE user_email = ? AND workspace_name = ?", (user_email_to_delete, workspace_name_to_delete))
            # Delete annotations
            cursor.execute("DELETE FROM annotations WHERE user_email = ? AND workspace_name = ?", (user_email_to_delete, workspace_name_to_delete))
            # Delete datasets
            cursor.execute("DELETE FROM datasets WHERE user_email = ? AND workspace_name = ?", (user_email_to_delete, workspace_name_to_delete))
            # Delete the workspace record itself
            cursor.execute("DELETE FROM workspaces WHERE user_email = ? AND workspace_name = ?", (user_email_to_delete, workspace_name_to_delete))
            
            # The 'with' statement handles conn_local.commit() on success and conn_local.close() automatically.
            
            st.success(f"✅ Successfully deleted workspace and all associated data for: {workspace_name_to_delete}")
            
    except Exception as e:
        st.error(f"Failed to delete workspace {workspace_name_to_delete}: Database Error: {e}")
        return
        
    # CRITICAL: Clear cache and rerun to refresh the page display
    try:
        st.cache_resource.clear()
        st.cache_data.clear() 
    except Exception:
        pass

    st.rerun()
def fetch_dataset_blob_for_admin(user_email, workspace_name):
    """Retrieves the filename and binary data (BLOB) for a specific dataset using a safe, local connection."""
    
    filename, data_bytes = None, None
    
    try:
        # CRITICAL FIX: Use a new, local connection context manager for safety
        with sqlite3.connect("users.db", check_same_thread=False) as conn_local:
            local_cursor = conn_local.cursor()
            
            local_cursor.execute(
                "SELECT filename, data FROM datasets WHERE user_email=? AND workspace_name=?", 
                (user_email, workspace_name)
            )
            result = local_cursor.fetchone()
            
            if result:
                filename, data_bytes = result[0], result[1]
                
        return filename, data_bytes
        
    except Exception as e:
        # Use a print statement instead of st.error here to avoid disrupting the layout 
        # during the column rendering loop, but the logic should catch DB errors safely.
        print(f"Error fetching dataset BLOB for admin: {e}")
        return None, None
def log_activity(user_email, event_type, details=""):
    """Logs user activity (LOGIN, LOGOUT, SESSION_START, SESSION_END) using UTC time."""
    global conn
    
    # Generate explicit UTC timestamp string
    utc_timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    try:
        # Note: We must explicitly list 'timestamp' in the query now
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO activity_log (user_email, event_type, timestamp, details) 
            VALUES (?, ?, ?, ?)
            """,
            (user_email, event_type, utc_timestamp, details) # Pass UTC timestamp
        )
        conn.commit()
    except Exception as e:
        # If the table doesn't exist yet, try to create it, then try logging again
        if "no such table: activity_log" in str(e):
            # Assumes initialize_database_schema is available
            initialize_database_schema() 
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO activity_log (user_email, event_type, timestamp, details) 
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_email, event_type, utc_timestamp, details)
                )
                conn.commit()
            except Exception as e2:
                print(f"Failed to log activity after schema init: {e2}")
        else:
            print(f"Error logging activity: {e}")
# ===============================
# DATA FETCHING UTILITY FOR ANNOTATION
# ===============================


@st.cache_data(show_spinner=False, ttl=300)
def get_sentences_for_annotation(user_email, workspace_name):
    """
    Retrieves the dataset from DB, reads the CSV, extracts sentences,
    and joins them with existing annotations. Returns a DataFrame of sentences.
    """
    try:
        # 1. Fetch the dataset (CSV file content) from the 'datasets' table
        local_cursor = conn.cursor()
        local_cursor.execute(
            "SELECT data FROM datasets WHERE user_email=? AND workspace_name=?",
            (user_email, workspace_name)
        )
        result = local_cursor.fetchone()
        
        if not result:
            return pd.DataFrame()

        # Extract the binary data (BLOB)
        file_data_bytes = result[0]
        
        # 2. Read the binary data into a pandas DataFrame
        df = pd.read_csv(BytesIO(file_data_bytes))

        # 3. Standardize the sentence column name.
        # We check for common column names: 'text', 'sentence', or 'utterance'.
        sentence_col = None
        for col in ['text', 'sentence', 'utterance', 'Input', 'user_statement']:
            if col in df.columns:
                sentence_col = col
                break
        
        if not sentence_col:
            st.warning("Could not find a sentence column ('text', 'sentence', or 'utterance') in the uploaded CSV.")
            return pd.DataFrame()

        # Rename the found column to a standard 'sentence'
        df = df.rename(columns={sentence_col: 'sentence'})
        
        # Select only the sentence column and drop any duplicates
        sentences_df = df[['sentence']].drop_duplicates().reset_index(drop=True)
        sentences_df['sentence_id'] = sentences_df.index # Create a unique ID for indexing

        # 4. Fetch existing annotations to mark data as annotated
        annotations_df = pd.read_sql(
            "SELECT sentence_id FROM annotations WHERE user_email=? AND workspace_name=?", 
            conn, 
            params=(user_email, workspace_name)
        )

        # 5. Join to determine annotated status (0=not annotated, 1=annotated)
        sentences_df = sentences_df.merge(
            annotations_df, 
            on='sentence_id', 
            how='left', 
            indicator='is_annotated'
        )
        sentences_df['is_annotated'] = sentences_df['is_annotated'].apply(lambda x: 1 if x == 'both' else 0)

        return sentences_df
        
    except Exception as e:
        st.error(f"Error loading sentences for annotation: {e}")
        return pd.DataFrame()
    
import pandas as pd
import streamlit as st

# --- Helper Function to Fetch History for ALL Workspaces ---
def get_all_workspace_history(user_email):
    """
    Retrieves all model evaluation history across all workspaces for the user.
    """
    local_cursor = conn.cursor()
    query = """
    SELECT workspace_name, version, model_name, accuracy, f1_score, train_date, trained_samples
    FROM models
    WHERE user_email = ?
    ORDER BY train_date DESC
    """
    try:
        local_cursor.execute(query, (user_email,))
        rows = local_cursor.fetchall()
        df = pd.DataFrame(rows, columns=[
            'Workspace', 
            'Version', 
            'Model Name',       
            'Accuracy', 
            'F1 Score',         
            'Train Date',       
            'Trained Samples'
        ])
        return df
    except Exception as e:
        # st.error(f"Error fetching all model history: {e}") 
        return pd.DataFrame()


# --- Page Function for Global Comparison ---
def show_workspace_comparison_tab():
    user_email = st.session_state.get('logged_in_email')
    
    if not user_email:
        st.error("User not logged in.")
        return

    st.subheader("Workspace vs. Workspace Comparison")
    st.info("Compare the best performing model (highest accuracy) from two different workspaces.")

    # 1. Fetch all model history
    all_history_df = get_all_workspace_history(user_email)
    
    if all_history_df.empty:
        st.warning("No model evaluation history found across any of your workspaces.")
        return

    # 2. Find the BEST model (highest accuracy) for each unique workspace
    # Group by workspace and find the index of the max accuracy for each group
    idx = all_history_df.groupby('Workspace')['Accuracy'].idxmax()
    best_models_df = all_history_df.loc[idx].reset_index(drop=True)
    
    workspace_options = ['-- Select Workspace --'] + best_models_df['Workspace'].tolist()
    
    if len(workspace_options) < 3: # Need at least 2 real workspaces + the default option
        st.warning(f"Found only {len(workspace_options) - 1} workspace(s) with trained models. Train more models to compare.")
        return

    # 3. Selection Dropdowns
    col1, col2 = st.columns(2)
    
    with col1:
        ws_a = st.selectbox("Select Workspace A (Baseline)", options=workspace_options, key="compare_ws_a")
    
    with col2:
        ws_b = st.selectbox("Select Workspace B (Candidate)", options=workspace_options, key="compare_ws_b")

    # 4. Perform Comparison
    if ws_a != '-- Select Workspace --' and ws_b != '-- Select Workspace --':
        if ws_a == ws_b:
            st.error("Please select two different workspaces for comparison.")
            return

        # Get the best metrics for the selected workspaces
        metrics_a = best_models_df[best_models_df['Workspace'] == ws_a].iloc[0]
        metrics_b = best_models_df[best_models_df['Workspace'] == ws_b].iloc[0]

        st.markdown("---")
        st.markdown(f"#### Comparison: **{ws_b}** vs. **{ws_a}**")

        # 5. Display Comparison Table
        comparison_data = {
            'Metric': ['Model Version', 'Accuracy', 'F1 Score', 'Trained Samples', 'Train Date'],
            ws_a: [
                metrics_a['Version'], 
                f"{metrics_a['Accuracy'] * 100:.2f}%", 
                f"{metrics_a['F1 Score'] * 100:.2f}%", 
                metrics_a['Trained Samples'], 
                metrics_a['Train Date']
            ],
            ws_b: [
                metrics_b['Version'], 
                f"{metrics_b['Accuracy'] * 100:.2f}%", 
                f"{metrics_b['F1 Score'] * 100:.2f}%", 
                metrics_b['Trained Samples'], 
                metrics_b['Train Date']
            ]
        }
        comparison_df = pd.DataFrame(comparison_data)

        st.dataframe(comparison_df, hide_index=True, use_container_width=True)

        # 6. Highlight Difference
        diff_acc = metrics_b['Accuracy'] - metrics_a['Accuracy']
        st.markdown("---")
        st.metric(
            label=f"Accuracy Difference ({ws_b} vs. {ws_a})", 
            value=f"{diff_acc * 100:.2f} percentage points", 
            delta=f"{diff_acc * 100:.2f}"
        )
    
def entities_list_to_input(entities_list):
    """Converts a list of entity dicts (e.g., [{'entity': 'artist', 'value': 'Monet'}])
    into the required comma-separated string format (e.g., 'artist:Monet, date:1872').
    """
    if not entities_list:
        return ""
    
    # Format entities as 'name:value' pairs
    entity_strings = [
        f"{d.get('entity', '').strip()}:{d.get('value', '').strip()}" 
        for d in entities_list 
        if d.get('entity') and d.get('value')
    ]
                      
    return ", ".join(entity_strings)
def set_page_state(page_name):
    """
    Sets the application's current page state (st.session_state.page) 
    and triggers a rerun to change the view.
    """
    st.session_state.page = page_name
   # st.rerun()
    
# --- SIMULATED TRAINING & EVALUATION FUNCTION ---

def _simulate_nlu_training_core(user_email, workspace_name, version, model_type, epochs, notes):
    """
    Simulates NLU model training, calculates metrics, and saves the result to the 'models' table.
    Includes a retry loop for SQLite database locking errors.
    """
    
    # 1. GET SAMPLE COUNT
    try:
        local_cursor_count = conn.cursor()
        local_cursor_count.execute(
            "SELECT COUNT(*) FROM annotations WHERE user_email = ? AND workspace_name = ?",
            (user_email, workspace_name)
        )
        sample_count = local_cursor_count.fetchone()[0]
    except Exception as e:
        st.error(f"Error reading sample count: {e}")
        return None, None, None
        
    if sample_count < 10:
        return None, None, sample_count

    # 2. SIMULATE METRICS (Calculated based on inputs)
    base_acc = 0.70
    base_f1 = 0.65
    
    # Adjust metrics for Transformer (trf) model and higher epochs
    if "trf" in model_type:
        base_acc += 0.10
        base_f1 += 0.10
    
    epoch_factor = epochs / 100.0
    
    accuracy = min(1.0, base_acc + (random.random() * 0.05 * epoch_factor))
    f1_score = min(1.0, base_f1 + (random.random() * 0.05 * epoch_factor))
    
    # Combine epochs and notes since the existing 'models' table lacks an 'epochs' column
    full_notes = f"Epochs: {epochs}. {notes}"


    # 3. SAVE TO DB (WITH RETRY LOOP) - Using the 'models' table
    max_retries = 5
    retry_delay = 0.5 
    
    for attempt in range(max_retries):
        try:
            local_cursor_save = conn.cursor()
            
            # Check for version existence before saving (Read 2) - Use 'models'
            local_cursor_save.execute(
                "SELECT version FROM models WHERE user_email=? AND workspace_name=? AND version=?",
                (user_email, workspace_name, version)
            )
            if local_cursor_save.fetchone():
                st.error(f"Version '{version}' already exists. Please choose a new version name.")
                return None, None, sample_count 

            # Insert new model history (Write 1) - Use 'models' and correct column names
            # model_type maps to model_name, sample_count maps to trained_samples
            local_cursor_save.execute(
                """
                INSERT INTO models (user_email, workspace_name, version, model_name, trained_samples, accuracy, f1_score, notes) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                # Note: train_date uses its default CURRENT_TIMESTAMP
                (user_email, workspace_name, version, model_type, sample_count, accuracy, f1_score, full_notes)
            )
            
            # Update workspace's last trained model info (Write 2)
            local_cursor_save.execute(
                """
                UPDATE workspaces SET last_model_version=?, last_trained_at=CURRENT_TIMESTAMP, last_model_type=? 
                WHERE user_email=? AND workspace_name=?
                """,
                (version, model_type, user_email, workspace_name)
            )
            
            conn.commit()
            
            # Success!
            return accuracy, f1_score, sample_count
            
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                if attempt < max_retries - 1:
                    print(f"Database locked, retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5 
                else:
                    st.error("Error saving model metrics: database is locked. Please try again.")
                    return None, None, sample_count
            else:
                st.error(f"Error during training simulation: {e}")
                return None, None, None
        
        except Exception as e:
            st.error(f"Unexpected error during training simulation: {e}")
            return None, None, None
# --- ACTIVE LEARNING UTILITIES ---

def predict_with_confidence(sentence, domain):
    """
    Runs the standard NLU prediction (Intent + Entities) and simulates a confidence score.
    
    Args:
        sentence (str): The sentence to predict.
        domain (str): The workspace domain.
        
    Returns:
        tuple: (predicted_intent, entities_json, confidence_score)
    """
    # Use your existing prediction logic
    predicted_intent, entities_json = predict_intent_and_entities(sentence, domain)
    
    # Simulate Confidence Score (Lower for 'default_fallback' or generic intents)
    if "default_fallback" in predicted_intent or "general_query" in predicted_intent or not predicted_intent:
        # 35% - 55% confidence: Mark as uncertain
        confidence = random.uniform(0.35, 0.55) 
    elif "greeting" in predicted_intent:
        # 95% - 99% confidence: Mark as certain
        confidence = random.uniform(0.95, 0.99) 
    else:
        # 60% - 90% confidence: Mark as normal/certain
        confidence = random.uniform(0.60, 0.90) 
        
    return predicted_intent, entities_json, confidence
# ==============================
# ENTITY MANAGEMENT UTILITY (FIXED for 'entity' key)
# ==============================
def entities_input_to_dict(entity_input_string):
    """Converts the simple k:v, k:v string (used for pre-loading) to a list of {entity: k, value: v} dicts."""
    entities_list = []
    if not entity_input_string:
        return entities_list
    
    try:
        parts = entity_input_string.split(',')
        for part in parts:
            if ':' in part:
                k, v = part.split(':', 1)
                # FIX: Use 'entity' key to match the dynamic editor functions
                entities_list.append({"entity": k.strip(), "value": v.strip()}) 
    except Exception:
        # Fail gracefully
        pass
        
    return entities_list

def dict_to_entities_json_string(entity_list):
    """Converts a list of {entity: k, value: v} dicts (from session state) back to the DB-ready JSON string."""
    entities_dict = {}
    for entity in entity_list:
        # FIX: Ensure both 'entity' and 'value' are present and not empty before saving
        if entity.get("entity", "").strip() and entity.get("value", "").strip(): 
            entities_dict[entity["entity"].strip()] = entity["value"].strip()
    return json.dumps(entities_dict)
# ==============================

# ==============================
# DYNAMIC ENTITY EDITOR FUNCTIONS (NEW - CORRECTED)
# ==============================

def add_new_entity():
    """Appends a new, empty entity dict to the session state list."""
    if 'current_entities' not in st.session_state or st.session_state.current_entities is None:
        st.session_state.current_entities = []
        
    st.session_state.current_entities.append({'entity': '', 'value': ''})

def remove_entity(index):
    """Removes an entity at the specified index."""
    if 'current_entities' in st.session_state and index < len(st.session_state.current_entities):
        st.session_state.current_entities.pop(index)

def update_entity(index, entity_key, widget_key):
    """
    Updates the 'entity' (Type) or 'value' (Value) of an entity dict by reading the 
    latest value from st.session_state using the widget's key (widget_key).
    """
    if 'current_entities' in st.session_state and index < len(st.session_state.current_entities):
        # Read the value that was just written by the st.text_input widget
        new_value = st.session_state[widget_key]
        st.session_state.current_entities[index][entity_key] = new_value

def render_dynamic_entity_editor(current_index):
    """
    Renders the dynamic entity editor shown in the screenshot.
    Entities are managed in st.session_state.current_entities.
    """
    
    st.markdown("### 2. Current Entities:")
    
    entities = st.session_state.current_entities
    
    # --- Header Row ---
    header_col1, header_col2, header_col3 = st.columns([4, 6, 0.5])
    with header_col1:
        st.markdown("**Type**")
    with header_col2:
        st.markdown("**Value**")
    
    st.markdown("---")

    # --- Dynamic Entity Rows ---
    for i, entity_data in enumerate(entities):
        col1, col2, col3 = st.columns([4, 6, 0.5])
        
        type_key = f"entity_type_{current_index}_{i}"
        with col1:
            st.text_input(
                label="Type", 
                value=entity_data.get('entity', ''), 
                key=type_key,
                label_visibility="collapsed",
                # CORRECTION: Pass the widget key name (type_key) as the third arg
                on_change=update_entity,
                args=(i, 'entity', type_key) 
            )
        
        value_key = f"entity_value_{current_index}_{i}"
        with col2:
            st.text_input(
                label="Value", 
                value=entity_data.get('value', ''), 
                key=value_key,
                label_visibility="collapsed",
                # CORRECTION: Pass the widget key name (value_key) as the third arg
                on_change=update_entity,
                args=(i, 'value', value_key) 
            )

        with col3:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            st.button(
                "❌", 
                key=f"remove_entity_{current_index}_{i}", 
                on_click=remove_entity, 
                args=(i,),
                type="secondary"
            )

    # --- Add New Entity Button ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.button(
        "➕ Add New Entity Row", 
        key=f"add_entity_btn_{current_index}", 
        on_click=add_new_entity
    )
    
    return entities
# ==============================

def get_existing_annotation(user_email, workspace_name, sentence):
    """Retrieves existing intent and entities for a given sentence from the DB. FIX: Uses entities_json."""
    local_cursor = conn.cursor()
    local_cursor.execute(
        # CRITICAL FIX: Use entities_json to match the table schema
        """SELECT intent, entities_json FROM annotations 
           WHERE user_email=? AND workspace_name=? AND sentence=?""",
        (user_email, workspace_name, sentence)
    )
    result = local_cursor.fetchone()
    # Returns (intent, entities_json_string) or (None, None)
    return result if result else (None, None)

def save_annotation_to_db(workspace_name, user_email, sentence, intent, entities_json, remarks=""):
    """Saves or updates the annotation using UPSERT (ON CONFLICT). FIX: Uses entities_json."""
    try:
        local_cursor = conn.cursor()
        # CRITICAL FIX: Include 'remarks' in INSERT and UPDATE
        local_cursor.execute(
            """
            INSERT INTO annotations (workspace_name, user_email, sentence, intent, entities_json, remarks, last_modified) 
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(workspace_name, user_email, sentence) DO UPDATE SET
            intent = excluded.intent, 
            entities_json = excluded.entities_json,
            remarks = excluded.remarks,
            last_modified = CURRENT_TIMESTAMP
            """, 
            (workspace_name, user_email, sentence, intent, entities_json, remarks)
        )
        conn.commit()
        return True
    except Exception as e:
        # Assuming st is defined globally
        st.error(f"Failed to save annotation to database: {e}") 
        return False
# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(page_title="BuddyBot", page_icon="🤖", layout="wide")

# ==============================
# DATABASE SETUP (THREAD-SAFE CONNECTION & MIGRATION)
# ==============================
# ==============================
# DATABASE SETUP (THREAD-SAFE CONNECTION & MIGRATION)
# ==============================
@st.cache_resource
def get_db_connection():
    """Creates and caches the SQLite connection object (Do NOT close this connection)."""
    conn = sqlite3.connect("users.db", check_same_thread=False)
    # Set to IMMEDIATE to handle schema creation robustly
    conn.isolation_level = None 
    return conn

# Global connection object (used by all functions)
conn = get_db_connection()


# Function to ensure all tables exist and handle migrations
def initialize_database_schema():
    global conn
    try:
        # Check if the connection is usable
        conn.cursor() 
    except sqlite3.ProgrammingError:
        # Connection is closed, force re-creation
        st.cache_resource.clear()
        conn = get_db_connection()
        #st.rerun()
        return

    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)

    # 2. Workspaces Table
    # NOTE: The migration logic (ALTER TABLE) is needed below the initial creation.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            workspace_name TEXT, 
            domain TEXT,
            last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            -- Columns added via migration below for older databases
            last_model_version TEXT, 
            last_model_type TEXT,
            last_trained_at TIMESTAMP,
            UNIQUE (user_email, workspace_name) 
        )
    """)

    # 3. Datasets Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_name TEXT, 
            user_email TEXT,
            filename TEXT,
            data BLOB,
            UNIQUE (user_email, workspace_name) 
        )
    """)

    # 4. Annotations Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            user_email TEXT,
            workspace_name TEXT,
            sentence TEXT,
            intent TEXT,
            entities_json TEXT,
            last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
            PRIMARY KEY (user_email, workspace_name, sentence)
        )
    """)
    
    # ANNOTATIONS MIGRATION (Add remarks column for feedback)
    try:
        cursor.execute("ALTER TABLE annotations ADD COLUMN remarks TEXT;")
    except sqlite3.OperationalError:
        pass

    # 5. Models Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            workspace_name TEXT NOT NULL, 
            version TEXT NOT NULL,           
            model_name TEXT,              
            train_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accuracy REAL,
            f1_score REAL,
            trained_samples INTEGER,      
            notes TEXT,
            UNIQUE (user_email, workspace_name, version)
        )
    """)
    # 6. Activity Log Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            details TEXT
        )
    """)
    
    # -------------------------------------------------------------------
    # SCHEMA MIGRATION: Handles missing columns if database existed before
    # -------------------------------------------------------------------

    # WORKSPACES MIGRATION (Fixes 'last_model_version' errors)
    try:
        cursor.execute("ALTER TABLE workspaces ADD COLUMN last_model_version TEXT;")
    except sqlite3.OperationalError:
        pass 
    try:
        cursor.execute("ALTER TABLE workspaces ADD COLUMN last_model_type TEXT;")
    except sqlite3.OperationalError:
        pass 
    try:
        cursor.execute("ALTER TABLE workspaces ADD COLUMN last_trained_at TIMESTAMP;")
    except sqlite3.OperationalError:
        pass 
        
    # MODELS MIGRATION (Skipping detailed checks, relying on your original comprehensive alter block)
    # If any error persists, delete the users.db file to ensure a clean start based on the CREATE TABLEs above.
    
    # Commit the changes (CREATE TABLE and ALTER TABLE operations)
    conn.commit()


# RUN THE INITIALIZATION
initialize_database_schema()
# ==============================
# DOMAIN DATA & NLU INTENTS
# ==============================
DOMAINS = {
    "Sports": {
        "icon": "⚽",
        "description": "Analyze team stats, game history, and player performance.",
        "intents": ["request_score", "query_player_stat", "book_ticket", "greeting"]
    },
    "Education": {
        "icon": "📚",
        "description": "Create learning assistants from textbooks, notes, or research papers.",
        "intents": ["query_definition", "request_summary", "schedule_study", "greeting"]
    },
    "Art & Design": {
        "icon": "🎨",
        "description": "Interpret artistic styles, history, or design principles.",
        "intents": ["query_artist", "describe_style", "find_gallery", "greeting"]
    },
    "Entertainment": {
        "icon": "🎬",
        "description": "Discuss movies, music, celebrities, and pop culture trends.",
        "intents": ["recommend_movie", "query_actor", "buy_merch", "greeting"]
    },
    "Finance": {
        "icon": "💰",
        "description": "Manage bank account queries, fund transfers, and fraud reporting.",
        "intents": ["query_balance", "transfer_funds", "report_fraud", "greeting"]
    },
    "Travel & Booking": {
        "icon": "✈️",
        "description": "Handle flight, hotel, and general ticket reservations and inquiries.",
        "intents": ["book_flight", "check_inquiry", "cancel_reservation", "greeting"]
    },
    "Business": {
        "icon": "💼",
        "description": "Handle general customer FAQs, submit help desk tickets, or analyze operations.",
        "intents": ["query_hours", "submit_complaint", "request_report", "greeting"]
    },
    "Healthcare": {
        "icon": "🏥",
        "description": "Provide information on symptoms, book appointments, or answer medical FAQs.",
        "intents": ["book_appointment", "query_symptom", "refill_prescription", "greeting"]
    },
    "IT Support": {
        "icon": "💻",
        "description": "Assist with troubleshooting, password resets, and software installation guides.",
        "intents": ["reset_password", "troubleshoot_login", "request_software", "greeting"]
    },
    "Real Estate": {
        "icon": "🏠",
        "description": "Search property listings, schedule viewings, and answer mortgage questions.",
        "intents": ["search_property", "schedule_viewing", "query_mortgage", "greeting"]
    },
    "E-commerce": {
        "icon": "🛒",
        "description": "Track orders, process returns, and handle product inventory questions.",
        "intents": ["track_order", "process_return", "query_inventory", "greeting"]
    }
}

# ==============================
# PAGE STYLING (Embedded CSS)
# ==============================
st.markdown("""
    <style>
        /* Base Styling */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0b1a37 0%, #1c2b4d 100%);
            color: white;
            animation: fadeIn 1s ease-in-out;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: 800;
            color: #6EC6FF;
            margin-bottom: 20px;
        }
        
        /* Sidebar Bot Avatar */
        .sidebar-bot-avatar {
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar-bot-avatar img {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background-color: #6EC6FF;
            padding: 5px;
            box-shadow: 0 0 10px rgba(110, 198, 255, 0.5);
        }
        .sidebar-bot-avatar p {
            font-size: 18px;
            font-weight: bold;
            color: white;
            margin-top: 10px;
        }
            
        /* Logo & Chat Bubbles */
        .logo-container {
            text-align: center;
            margin-top: 30px;
            animation: slideUp 1s ease-in-out;
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .chat-bubble {
            background-color: #2b70f0;
            border-radius: 15px;
            padding: 10px 15px;
            display: inline-block;
            margin: 10px 0;
            color: white;
            max-width: 90%;
            text-align: left;
        }
        .chat-bubble-container {
            margin-top: 20px;
            width: 100%;
            text-align: center;
        }

        /* Input & Auth Button Styles */
        .stTextInput > div > div > input, .stSelectbox > div > div > div > input {
            background-color: #1c2b4d;
            border: 1px solid #334466;
            color: white;
            transition: all 0.2s ease;
        }
        .stTextInput > div > div > input:focus, .stTextInput > div > div > input:hover {
            border-color: #6EC6FF !important;
            box-shadow: 0 0 0 1px #6EC6FF;
        }
        div.stButton button[kind="primary"] {
            background-color: white !important;
            color: #2b70f0 !important;
            border: 1px solid #2b70f0;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            transition: all 0.3s ease-in-out;
        }
        div.stButton button[kind="primary"]:hover {
            background-color: #f0f0f0 !important;
            transform: scale(1.02);
            border-color: #1748b0;
        }
        
        /* Custom Styling for Simple Domain Cards/Buttons */
        .domain-card {
            background-color: #1c2b4d;
            border: 2px solid #334466;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            text-align: center;
            /* Ensure fixed height for 4-column layout consistency */
            height: 150px; 
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .domain-card:hover {
            border-color: #6EC6FF;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5);
            transform: translateY(-3px);
        }
        .domain-card h3 {
            color: #6EC6FF;
            margin-top: 0;
            font-size: 20px;
        }
        .domain-card p {
            font-size: 14px;
            color: #ccc;
        }
        
        /* Sidebar styling for better appearance */
        [data-testid="stSidebar"] {
            background-color: #1c2b4d !important;
            color: white;
        }
        
        /* New: Entity Tagging Style Simulation */
        .sentence-display {
            font-size: 1.2em;
            padding: 15px;
            margin: 10px 0;
            background-color: #1c2b4d;
            border-radius: 8px;
            border: 1px solid #334466;
            min-height: 50px;
        }
        .status-card {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            color: white;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
        }
        .status-card.red { background-color: #9c3f3f; border-left: 5px solid #ff0000; } /* Critical */
        .status-card.yellow { background-color: #9c843f; border-left: 5px solid #ffbf00; } /* Warning */
        .status-card.green { background-color: #4b8d4b; border-left: 5px solid #00ff00; } /* Healthy */
        .status-card h5 { margin-top: 0; color: white; }
        .big-value { font-size: 2.5em; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==============================
# SESSION STATE & NAVIGATION HELPERS
# ==============================
if 'page' not in st.session_state:
    st.session_state.page = 'register'
if 'logged_in_email' not in st.session_state:
    st.session_state.logged_in_email = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_workspace' not in st.session_state:
    st.session_state.current_workspace = None
if 'current_domain' not in st.session_state:
    st.session_state.current_domain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'temp_workspace_name' not in st.session_state:
    st.session_state.temp_workspace_name = ""
if 'workspace_action' not in st.session_state:
    st.session_state.workspace_action = None
if 'sentences_df' not in st.session_state: 
    st.session_state.sentences_df = None
if 'annotation_index' not in st.session_state: 
    st.session_state.annotation_index = 0
# --- NEW ENTITY STATE ---
if 'current_entities' not in st.session_state:
    st.session_state.current_entities = []
if 'last_annotation_index' not in st.session_state:
    st.session_state.last_annotation_index = -1
if 'test_sentences_list' not in st.session_state:
    st.session_state.test_sentences_list = []
if 'test_sentence_index' not in st.session_state:
    st.session_state.test_sentence_index = 0
if 'al_queue' not in st.session_state:
    st.session_state.al_queue = None
if 'al_index' not in st.session_state:
    st.session_state.al_index = 0
def copy_query_to_remarks():
    """Copies sentence input to remarks if the general feedback box is checked."""
    # This logic runs when the checkbox state changes.
    if st.session_state.is_general_feedback_checkbox:
        # Check if the sentence field has content
        if st.session_state.feedback_sentence_input and st.session_state.feedback_sentence_input.strip():
            # Only copy if the remarks field is currently empty (to avoid overwriting user edits)
            if not st.session_state.feedback_remarks_input.strip():
                st.session_state.feedback_remarks_input = f"User Query: {st.session_state.feedback_sentence_input}"
# --- END NEW ENTITY STATE ---

# Check query parameters for initial navigation
if 'page' in st.query_params:
    st.session_state.page = st.query_params['page']

# Navigation Functions
def navigate_to_login():
    st.session_state.page = 'login'
    st.query_params['page'] = 'login'

def navigate_to_register():
    st.session_state.page = 'register'
    st.query_params['page'] = 'register'

def navigate_to_home():
    # --- NEW: Check if Admin is using the Home Navigation ---
    admin_email = "admin@example.com"
    if st.session_state.get('logged_in_email') == admin_email:
        st.session_state.page = 'admin'
        st.query_params['page'] = 'admin'
    else:
        st.session_state.page = 'home'
        st.query_params['page'] = 'home'

def navigate_to_create_workspace():
    st.session_state.page = 'create_workspace'
    st.query_params['page'] = 'create_workspace'

def navigate_to_admin():
    st.session_state.page = 'admin'
    st.query_params['page'] = 'admin'

def navigate_to_workspace():
    st.session_state.page = 'workspace'
    st.query_params['page'] = 'workspace'

def navigate_to_feedback_page():
    """Custom router for the new page."""
    st.session_state.page = 'feedback_module'
# CODE MODIFIED (Around line 105)
def navigate_to_action_choice():
    st.session_state.page = 'action_choice'
    st.query_params['page'] = 'action_choice'
    # CRITICAL FIX: Ensure no action is active when viewing the choice page
    st.session_state.workspace_action = None
    
def navigate_to_annotate(): 
    st.session_state.page = 'annotate'
    st.query_params['page'] = 'annotate'

def navigate_to_evaluate(): # RE-ADDED EVALUATION NAVIGATION
    st.session_state.page = 'evaluate'
    st.query_params['page'] = 'evaluate'

def navigate_to_policy():
    st.session_state.page = 'policy'
    st.query_params['page'] = 'policy'
    
# Callback function for domain selection
def finalize_workspace_creation(workspace_name, domain_name):
    """Callback for creating and activating a new workspace."""
    local_cursor = conn.cursor() 
    
    try:
        local_cursor.execute(
            "INSERT INTO workspaces (user_email, workspace_name, domain) VALUES (?, ?, ?)",
            (st.session_state.logged_in_email, workspace_name, domain_name)
        )
        conn.commit()
        
        st.session_state.current_workspace = workspace_name
        st.session_state.current_domain = domain_name
        st.session_state.messages.clear()
        st.session_state.temp_workspace_name = "" 
        
        navigate_to_action_choice()
    except sqlite3.IntegrityError:
        st.error("⚠️ A workspace with that name already exists. Please choose a different name.")
        
# Callback for activating existing workspace
def activate_existing_workspace(workspace_name, domain_name):
    """Callback for setting an existing workspace as active."""
    st.session_state.current_workspace = workspace_name
    st.session_state.current_domain = domain_name
    
    if workspace_name not in st.session_state.chat_history:
        st.session_state.chat_history[workspace_name] = []
        
    st.session_state.messages = st.session_state.chat_history.get(workspace_name, [])
    
    navigate_to_action_choice()

# Callback to set action and move to workspace page
def set_workspace_action(action):
    """Callback for setting the current action (Train, Test, Annotate, Evaluate)."""
    st.session_state.workspace_action = action
    
    if action == "Annotate":
        navigate_to_annotate()
    elif action == "Evaluate": # NEW: Evaluate action
        navigate_to_evaluate()
    else:
        navigate_to_workspace()
def go_to_next_sentence():
    """Increments the sentence index and clears results to trigger a new prediction."""
    st.session_state.test_sentence_index += 1
    # Clear prediction state to force a re-run of the prediction logic on the new sentence
    st.session_state.last_prediction_results = None 
    st.session_state['correct_intent_select'] = '-- Select --' # Reset feedback selection

def save_feedback_and_next(sentence, predicted_intent, entities_json, is_correct, user_email, workspace_name, correct_intent=None, remarks=""):
    """Saves the feedback and moves to the next sentence."""
    
    # Use existing utility to save or update the annotation
    # PASS REMARKS HERE:
    if save_annotation_to_db(workspace_name, user_email, sentence, correct_intent if not is_correct else predicted_intent, entities_json, remarks):
        if is_correct:
            st.toast(f"✅ Saved correct intent: {predicted_intent}", icon="💾")
        else:
            st.toast(f"✍️ Saved correction: {correct_intent}", icon="💾")
        
        # Move to the next sentence
        go_to_next_sentence()
        st.rerun() # Trigger the next sentence display
    else:
        st.error("Error saving feedback to database.")

def skip_and_next():
    """Skips the current sentence without saving and moves to the next."""
    st.toast("Skipped sentence.", icon="⏭️")
    go_to_next_sentence()
    st.rerun()

# ==============================
# SIDEBAR CONTENT FUNCTION
# ==============================
def show_sidebar_content():
    # Show sidebar on ALL pages EXCEPT the 'register' page
    if st.session_state.page == 'register':
        return

    user_email = st.session_state.logged_in_email
    workspace = st.session_state.current_workspace 
    
    st.sidebar.markdown("""
        <div class="sidebar-bot-avatar">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712100.png" alt="Bot Avatar">
            <p>BuddyBot</p>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Check if the user is logged in to display user/workspace specific details
    if user_email:
        admin_email = "admin@example.com" # Define the admin email here too

        #if user_email == admin_email:
            # Only show button if the user is the admin
         #   if st.sidebar.button("⚙️ Admin Portal", key="nav_admin"): 
          #      st.session_state.page = 'admin'
        #st.sidebar.markdown(f"**Current Workspace:** `{workspace}`")
        #st.sidebar.markdown(f"**User:** `{user_email}`")
        st.sidebar.markdown(f"**Current Workspace:** `{workspace}`")
        
        # Logout button 
        st.sidebar.button("Logout", key="logout_sidebar", use_container_width=True, on_click=lambda: (
            # --- LOGOUT LOGIC UPDATED ---
            log_activity(user_email, "LOGOUT", "Session ended."), 
            
            st.session_state.pop('logged_in_email', None), 
            st.session_state.pop('current_workspace', None), 
            st.session_state.pop('current_domain', None), 
            st.session_state.messages.clear(), 
            st.session_state.pop('chat_history', None), 
            navigate_to_login()
        ))
        st.sidebar.markdown("---")

        with st.sidebar.expander("❓ **Help Section**"):
            st.markdown("1. **Select/Create Workspace** to specialize the bot.")
            st.markdown("2. **Upload a CSV** to train the bot.")
            st.markdown("3. **Annotate** the data to teach the NLU model.") 
            st.markdown("4. **Train** the model then **Chat**.") 
            st.markdown("5. **Evaluate** model versions.") 
        st.sidebar.markdown("---")

        if workspace:
            st.sidebar.markdown(f"**Chat History: {workspace}**") 
            if st.session_state.chat_history.get(workspace):
                with st.sidebar.container(height=200):
                    for i, msg in enumerate(st.session_state.chat_history.get(workspace, [])): 
                        if msg["role"] == "user":
                            summary = msg["content"][:30] + "..." if len(msg["content"]) > 30 else msg["content"]
                            st.sidebar.markdown(f"*{i+1}. {summary}*")
            else:
                st.sidebar.markdown("No history for this workspace yet.")
        else:
            st.sidebar.markdown("**Activate a workspace to view chat history.**")
        st.sidebar.markdown("---")

    else:
        # Simple message for non-logged-in users on login/policy pages
        st.sidebar.markdown("""
            <p style='color:#ccc; text-align:center;'>Please log in to access your workspaces and features.</p>
        """, unsafe_allow_html=True)

# ==============================
# DATA LOADERS/HANDLERS
# ==============================

@st.cache_data
# FIX 1: ADD workspace_last_modified as a cache key
def load_dataset_blob(user_email, workspace_name, workspace_last_modified): 
    """Retrieves the dataset BLOB and converts it to a DataFrame."""
    # workspace_last_modified is used solely as a cache key for Streamlit
    local_cursor = conn.cursor()
    local_cursor.execute(
        "SELECT data FROM datasets WHERE user_email=? AND workspace_name=?", 
        (user_email, workspace_name)
    )
    result = local_cursor.fetchone()
    
    if result:
        data_blob = result[0]
        try:
            # Convert BLOB to BytesIO object, then read as CSV
            df = pd.read_csv(BytesIO(data_blob))
            return df
        except Exception as e:
            # Print error to console/logs for better debugging if loading fails
            print(f"Error reading data from DB: {e}") 
            st.error("Error reading data from DB. File format might be corrupted.")
            return None
    return None

def split_dataframe_to_sentences(df):
    """
    Splits the content of the primary text column into a list of sentences.
    """
    text_col = None
    
    # 1. Search for a preferred column name (case-insensitive for robustness)
    # >>> FIX APPLIED HERE: Added 'user_statement' to the list of recognized columns.
    for col in df.columns:
        if col.lower() in ['text', 'sentence', 'utterance', 'user_statement']:
            text_col = col
            break
    
    if text_col:
        # Simple splitting for demonstration
        # Ensure only string data is used for splitting
        all_text = " ".join(df[text_col].astype(str).tolist())
        # Basic sentence splitting by period
        sentences = [s.strip() for s in all_text.split('.') if s.strip()]
        return pd.DataFrame(sentences, columns=['sentence'])
    
    st.warning("Could not find a 'text', 'sentence', 'utterance', or 'user_statement' column in the uploaded dataset. Using all columns as one string.")
    # Fallback: concatenate all columns for a simple string split
    all_text = df.astype(str).agg(' '.join, axis=1).str.cat(sep=' ')
    sentences = [s.strip() for s in all_text.split('.') if s.strip()]
    return pd.DataFrame(sentences, columns=['sentence'])

# ==============================
# NLU MODEL INTEGRATION (STUBS)
# ==============================

@st.cache_resource
def get_spacy_model():
    """Load the spacy model once."""
    if spacy:
        try:
            # This is the small English model pre-trained for NER
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except OSError:
            st.error("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
            return None
    return None

def is_model_trained(workspace_name):
    """Simulates checking if a model has been trained for the workspace."""
    # We rely on a simple session state flag if a 'train' button was pressed.
    local_cursor = conn.cursor()
    local_cursor.execute(
        "SELECT COUNT(*) FROM annotations WHERE workspace_name=?", 
        (workspace_name,)
    )
    count = local_cursor.fetchone()[0]
    return count > 0 or st.session_state.get(f'trained_{workspace_name}', False)

def train_nlu_model(workspace_name, annotated_data):
    """Simulates the NLU model training process."""
    st.info(f"🚀 Training NLU Model for **{workspace_name}** with {len(annotated_data)} examples. This is a simulated process...")
    time.sleep(2) # Simulate training time
    
    # Set a flag to show the model is 'trained' for testing purposes
    st.session_state[f'trained_{workspace_name}'] = True
    
    # Log the simulated model save (no actual file save)
    local_cursor = conn.cursor()
    local_cursor.execute(
        "UPDATE workspaces SET last_modified=CURRENT_TIMESTAMP WHERE workspace_name=?",
        (workspace_name,)
    )
    conn.commit()
    st.success(f"✅ NLU Model training simulated successfully! Now ready to **Test**.")
    return True

# --- REVERTED PREDICT FUNCTION (using old simulated response) ---
def predict_intent_and_entities(prompt, domain):
    """
    Uses the loaded spaCy model for NER and a rule-based approach for intent prediction.
    CRITICAL FIX: Prioritizes looking up the prompt in the custom annotations table
    to simulate a model that has learned from the user's data.
    """
    p = prompt.strip()
    
    # --- 1. CUSTOM TRAINING LOOKUP (Simulate Trained Model) ---
    try:
        # Look for an exact match in the custom annotations table
        local_cursor = conn.cursor()
        local_cursor.execute(
            "SELECT intent, entities_json FROM annotations WHERE sentence=?",
            (p,)
        )
        result = local_cursor.fetchone()
        if result:
            # If an exact match is found, return the custom-labeled intent and entities
            return result[0], result[1] # (intent, entities_json)
    except Exception as e:
        # If the lookup fails (e.g., DB error), fall through to the generic prediction
        print(f"Annotation lookup failed in prediction: {e}")


    # --- 2. FALLBACK TO GENERIC NLU (Original Stub Logic) ---
    # Note: If no exact match is found in custom data, we fall back to generic prediction.
    
    nlp = get_spacy_model()
    p_lower = prompt.lower()
    entities_dict = {}

    # Fallback to the old simulation if spacy is not available
    if not nlp:
        # st.warning("SpaCy model not loaded. Falling back to keyword prediction.") # Removed warning for cleaner UI
        
        # Travel & Booking Fallback
        if "book" in p_lower or "tickets" in p_lower or "reservation" in p_lower:
            intent = "book_flight" if "flight" in p_lower or "plane" in p_lower else "book_ticket"
            destination = "unknown"
            date = "today"
            # Simple Entity structure for fallback
            entities = json.dumps({"destination": destination, "type": intent.split('_')[1], "date": date})
            return intent, entities
        
        # Other simple fallbacks
        if "hello" in p_lower or "hi" in p_lower: return "greeting", "{}"
        if "smash" in p_lower or "drop shot" in p_lower or "improve" in p_lower or "drills" in p_lower: return "Training Tips", entities_json
        if "tournament" in p_lower or "open" in p_lower or "championship" in p_lower or "results" in p_lower: return "Tournament Info", entities_json
        if "service" in p_lower or "fault" in p_lower or "net" in p_lower or "rules" in p_lower: return "Rules", entities_json
        return "default_fallback", "{}"

    # --- SPACY NLU EXECUTION ---
    doc = nlp(prompt)
    
    # 1. Entity Extraction using spaCy's pre-trained NER
    for ent in doc.ents:
        entity_name = ent.label_.lower()
        entities_dict[entity_name] = ent.text
    entities_json = json.dumps(entities_dict)

    # 2. Rule-Based Intent Classification
    if "hello" in p_lower or "hi" in p_lower: 
        return "greeting", entities_json
    if any(key in p_lower for key in ["book", "flight", "ticket", "reserve"]):
        return "book_flight", entities_json
    if any(key in p_lower for key in ["balance", "funds", "transfer"]):
        return "query_balance", entities_json
    if any(key in p_lower for key in ["annotate", "label", "tag"]):
        return "meta_query_training", entities_json 
    if "smash" in p_lower or "drop shot" in p_lower or "improve" in p_lower or "drills" in p_lower:
        return "Training Tips", entities_json
    if "tournament" in p_lower or "open" in p_lower or "championship" in p_lower or "results" in p_lower:
        return "Tournament Info", entities_json
    if "service" in p_lower or "fault" in p_lower or "net" in p_lower or "rules" in p_lower:
        return "Rules", entities_json
        
    return "default_fallback", entities_json
# ==============================
def get_model_history(user_email, workspace_name):
    """Retrieves all model evaluation history, returning an empty DF on error."""
    # NOTE: This function relies on the global 'conn' object.
    local_cursor = conn.cursor()
    
    query = """
    SELECT version, model_name, accuracy, f1_score, trained_samples, train_date, notes
    FROM models
    WHERE user_email = ? AND workspace_name = ?
    ORDER BY train_date DESC
    """
    
    try:
        local_cursor.execute(query, (user_email, workspace_name))
        rows = local_cursor.fetchall()
        
        # CRITICAL FIX: Ensure column names match the casing/spacing used in show_evaluation_page.
        df = pd.DataFrame(rows, columns=[
            'Version', 
            'Model Name',       # Corrected from 'Model'
            'Accuracy', 
            'F1 Score',         # Corrected from 'F1' in previous iterations
            'Trained Samples',  # Corrected from 'Trained On'
            'Train Date',       # Corrected from 'Date'
            'Notes'
        ])
        return df
    except sqlite3.OperationalError:
        # Fails gracefully if the table/columns are missing
        return pd.DataFrame() 
    except Exception:
        return pd.DataFrame()

def simulate_nlu_training_and_save(user_email, workspace_name, version, model_name, epochs, notes):
    """Simulates NLU model training and saves the resulting metrics to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT COUNT(*) FROM annotations WHERE user_email = ? AND workspace_name = ?",
        (user_email, workspace_name)
    )
    trained_samples = cursor.fetchone()[0]

    if trained_samples < 10:
        st.error(f"Cannot start training. You need at least 10 annotated samples. Found: {trained_samples}")
        return

    st.info(f"Starting training simulation for version **{version}** with **{trained_samples}** samples...")
    time.sleep(2) 
    
    # Simulated metrics (Higher for BERT/trf model)
    if "trf" in model_name.lower():
        base_accuracy = 0.94 
        base_f1 = 0.92
    else:
        base_accuracy = 0.82
        base_f1 = 0.79
        
    random_variation = random.uniform(-0.01, 0.01)
    final_accuracy = min(base_accuracy + random_variation, 0.99)
    final_f1 = min(base_f1 + random_variation, 0.98)
    
    st.success("✅ Training and Evaluation Complete (Simulated).")
    
    try:
        cursor.execute(
            """
            INSERT INTO models (user_email, workspace_name, version, model_name, accuracy, f1_score, trained_samples, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_email, workspace_name, version, model_name, final_accuracy, final_f1, trained_samples, notes)
        )
        conn.commit()
        st.success(f"Metrics saved successfully! **Version {version}** | Accuracy: {final_accuracy:.2f} | F1: {final_f1:.2f}")
        time.sleep(1)
        st.session_state.page = 'evaluate'
        st.rerun()
        
    except sqlite3.IntegrityError:
        st.error(f"Version '{version}' already exists. Choose a new name.")
    except Exception as e:
        st.error(f"Error saving model metrics: {e}")
   

# Dummy function to simulate running evaluation and saving a new model version
def simulate_evaluation_and_save(user_email, workspace_name, version, notes=""):
    """Simulates evaluation metrics and saves a new model version."""
    
    # Fetch current number of versions to determine simulated performance
    current_versions = len(get_model_history(user_email, workspace_name))
    
    # Initial baseline
    base_acc = 0.65
    base_f1 = 0.60
    
    # Simulate improvement with more versions
    accuracy = round(min(0.95, base_acc + current_versions * 0.05 + random.uniform(-0.02, 0.02)), 2)
    f1_score = round(min(0.92, base_f1 + current_versions * 0.05 + random.uniform(-0.02, 0.02)), 2)
    
    
    # 2. Save to DB
    try:
        local_cursor = conn.cursor()
        local_cursor.execute(
            """
            INSERT INTO models (user_email, workspace_name, version, accuracy, f1_score, notes) 
            VALUES (?, ?, ?, ?, ?, ?)
            """, 
            (user_email, workspace_name, version, accuracy, f1_score, notes)
        )
        conn.commit()
        return accuracy, f1_score
    except sqlite3.IntegrityError:
        st.error(f"Version '{version}' already exists. Please choose a new version name.")
        return None, None
    except Exception as e:
        st.error(f"Failed to save evaluation: {e}") 
        return None, None


# ==============================
# TEST PAGE (SINGLE-SHOT & BATCH)FUNCTIONS
# ==============================
def save_feedback_annotation(sentence, predicted_intent, entities_json, is_correct, user_email, workspace_name, correct_intent=None):
    """Saves the user feedback (correct or incorrect prediction) back to the annotations table."""
    
    # NOTE: The current sentence is extracted from st.session_state.test_input
    if not sentence:
        # Check if the sentence is present in the session state
        sentence = st.session_state.get('test_input', '').strip()
        if not sentence:
            st.error("Cannot save feedback: Sentence is missing. Please re-enter the test utterance.")
            return

    # If the user confirmed the prediction was correct, save the prediction as a good example
    if is_correct:
        intent_to_save = predicted_intent
        feedback_message = f"✅ Saved sentence with intent **{intent_to_save}** as correct annotation."
    
    # If the user marked it wrong, save the sentence with the user-provided correct_intent
    else:
        if not correct_intent or correct_intent == '-- Select --':
            return st.error("Please select the correct intent before submitting the 'Wrong' feedback.")
        intent_to_save = correct_intent
        feedback_message = f"✍️ Saved sentence with corrected intent **{intent_to_save}** for future training."

    # Use existing utility to save or update the annotation
    if save_annotation_to_db(workspace_name, user_email, sentence, intent_to_save, entities_json):
        st.success(feedback_message)
        # Clear single-shot state to refresh the display
        st.session_state.last_prediction_results = None
        # Must manually reset the selection for the next run
        st.session_state['correct_intent_select'] = '-- Select --' 
        st.rerun()

    else:
        st.error("Error saving feedback to database.")
# ==============================
# TEST PAGE (SINGLE-SHOT & INTERACTIVE BATCH) FUNCTIONS
# ==============================
def show_chat_test_page(workspace_name, user_email):
    """
    Handles the Test interface, supporting single-shot NLU prediction and 
    interactive, sentence-by-sentence batch testing. It now routes feedback
    to a separate page.
    """
    domain = st.session_state.current_domain
    domain_display = DOMAINS.get(domain, {}).get("icon", "") + " " + domain
    
    st.subheader(f"🧪 Test NLU Model - {domain_display} Domain")
    
    # 1. Check Model Readiness
    if not is_model_trained(workspace_name):
        st.warning("Please train your NLU model first before testing.")
        if st.button("Go to Train Data", key="go_to_train_from_test_page"):
            set_workspace_action("Train")
            st.rerun()
        return

    # Initialize session state variables for results
    if 'last_prediction_results' not in st.session_state:
        st.session_state.last_prediction_results = None
    if 'batch_results_df' not in st.session_state:
        st.session_state.batch_results_df = pd.DataFrame() 
    if 'correct_intent_select' not in st.session_state:
        st.session_state['correct_intent_select'] = '-- Select --'
    
    
    is_interactive_mode = bool(st.session_state.test_sentences_list)
    
    # ----------------------------------------------------
    # A. INTERACTIVE BATCH MODE HANDLING
    # ----------------------------------------------------
    if is_interactive_mode:
        total_sentences = len(st.session_state.test_sentences_list)
        current_index = st.session_state.test_sentence_index
        
        # Check if the batch is complete
        if current_index >= total_sentences:
            st.success("🎉 Interactive batch testing complete!")
            st.session_state.test_sentences_list = [] # Exit interactive mode
            st.session_state.test_sentence_index = 0
            return
            
        current_sentence = st.session_state.test_sentences_list[current_index]

        st.markdown("---")
        st.markdown(f"#### 📝 Interactive Annotation: Sentence **{current_index + 1}** of **{total_sentences}**")
        
        # Display the current sentence from the batch (disable input)
        st.text_area(
            "Sentence to Test:",
            value=current_sentence,
            height=70,
            disabled=True,
            key="interactive_test_input"
        )
        
        # Auto-run prediction for the new sentence
        if st.session_state.last_prediction_results is None or st.session_state.last_prediction_results.get('sentence') != current_sentence:
            # Run prediction only if it's a new sentence or results are empty
            intent, entities_json = predict_intent_and_entities(current_sentence, domain)
            
            try:
                entities_dict = json.loads(entities_json)
            except json.JSONDecodeError:
                entities_dict = {"Error": "Invalid JSON format"}

            confidence = 0.85 if intent != "default_fallback" else 0.55
            
            st.session_state.last_prediction_results = {
                'sentence': current_sentence,
                'intent': intent,
                'confidence': f"{confidence:.2f}",
                'entities': entities_dict
            }
            # Rerun to display the results in the next block
            st.rerun()

    # ----------------------------------------------------
    # B. SINGLE-SHOT MODE HANDLING
    # ----------------------------------------------------
    else: # Not in interactive mode (Original single-shot test)
        st.markdown("---")
        with st.form(key='nlu_test_form'):
            st.markdown("#### Single Utterance Test")
            prompt = st.text_input(
                "Enter your test utterance:", 
                key="test_input", 
                placeholder="e.g., I want to book a flight to London next week."
            )
            submit_button = st.form_submit_button(
                label='Predict Intent & Entities', 
                type="primary",
                use_container_width=True
            )

            if submit_button and prompt:
                # Clear batch results when running single-shot test
                st.session_state.batch_results_df = pd.DataFrame()
                
                intent, entities_json = predict_intent_and_entities(prompt, domain)
                
                try:
                    entities_dict = json.loads(entities_json)
                except json.JSONDecodeError:
                    entities_dict = {"Error": "Invalid JSON format"}

                confidence = 0.85 if intent != "default_fallback" else 0.55
                
                # Store prediction results
                st.session_state.last_prediction_results = {
                    'sentence': prompt, 
                    'intent': intent,
                    'confidence': f"{confidence:.2f}",
                    'entities': entities_dict
                }
                
                # --- AUTO-REDIRECT TO FEEDBACK PAGE (Single-Shot Mode) ---
                st.session_state.feedback_data = {
                    "sentence": prompt,
                    "predicted_intent": intent,
                    "entities": entities_dict,
                    "workspace_name": workspace_name,
                    "user_email": user_email,
                    "domain": domain,
                    "is_interactive_mode": False # Not in a batch loop
                }
                navigate_to_feedback_page()
                st.rerun()
                
    # ----------------------------------------------------
    # C. PREDICTION RESULTS DISPLAY (Common to both modes)
    # ----------------------------------------------------
    if st.session_state.last_prediction_results and st.session_state.batch_results_df.empty:
        results = st.session_state.last_prediction_results
        
        # Use the sentence saved in the prediction results (works for both modes)
        tested_sentence = results.get('sentence', st.session_state.get('test_input', '')).strip()

        if not is_interactive_mode:
            st.markdown("---")
            
        st.markdown("### NLU Prediction Results")

        col1, col2 = st.columns(2)
        
        col1.metric(label="Predicted Intent", value=results['intent'])
        col2.metric(label="Confidence Score", value=results['confidence'])

        st.markdown("#### Extracted Entities")
        
        if results['entities']:
            entity_data = {
                "Entity Type": list(results['entities'].keys()),
                "Extracted Value": list(results['entities'].values())
            }
            st.dataframe(
                pd.DataFrame(entity_data), 
                hide_index=True, 
                use_container_width=True
            )
        else:
            st.info("No entities were extracted for this utterance.")
            
        # ----------------------------------------------------
        # FEEDBACK BUTTON (Routes to separate page)
        # ----------------------------------------------------
        st.markdown("---")
        
        # Store necessary data for the feedback page navigation
        feedback_data = {
            "sentence": tested_sentence,
            "predicted_intent": results['intent'],
            "entities": results['entities'],
            "workspace_name": workspace_name,
            "user_email": user_email,
            "domain": domain,
            "is_interactive_mode": is_interactive_mode
        }
        
        # CRITICAL: This button routes the user to the correction module.
        if st.button("✍️ Provide Feedback / Correction", key="route_to_feedback", type="primary"):
            st.session_state.feedback_data = feedback_data
            navigate_to_feedback_page()
            st.rerun()

        # Only show skip button if in interactive mode
        if is_interactive_mode:
            if st.button("⏭️ Skip & Next", key="skip_next", type="secondary"):
                skip_and_next() # Assumed to call go_to_next_sentence()

    # ----------------------------------------------------
    # D. BATCH FILE UPLOAD AND INTERACTIVE START
    # ----------------------------------------------------
    if not is_interactive_mode:
        st.markdown("---")
        with st.expander("⬆️ **Start Interactive Testing: Upload Dataset**"):
            uploaded_file = st.file_uploader(
                "Choose a CSV file containing test utterances.",
                type=['csv'],
                key="test_csv_uploader"
            )

            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
                    test_df = pd.read_csv(uploaded_file)
                    
                    # Identify the column to use
                    text_col = None
                    for col in test_df.columns:
                        if col.lower() in ['text', 'sentence', 'utterance', 'user_statement', 'input', 'user input']:
                            text_col = col
                            break
                    
                    if not text_col:
                        st.error("Could not find a 'text', 'sentence', or 'utterance' column. Please rename a column in your CSV.")
                    else:
                        st.dataframe(test_df.head(5), use_container_width=True)
                        
                        sentences_list = test_df[text_col].astype(str).tolist()
                        valid_sentences = [s.strip() for s in sentences_list if s.strip() and s.lower() != 'nan']
                        
                        st.info(f"File loaded with **{len(valid_sentences)}** sentences ready for interactive testing.")
                        
                        if st.button("Start Interactive Testing", key="start_batch_btn", type="primary", use_container_width=True):
                            if valid_sentences:
                                st.session_state.test_sentences_list = valid_sentences
                                st.session_state.test_sentence_index = 0
                                st.session_state.last_prediction_results = None 
                                st.session_state.batch_results_df = pd.DataFrame() # Clear old batch results
                                st.rerun()
                            else:
                                st.error("No valid sentences found in the selected column.")

                except Exception as e:
                    st.error(f"Error processing uploaded file: {e}")

    # ----------------------------------------------------
    # E. BACK BUTTON
    # ----------------------------------------------------
    st.markdown("---")
    if st.button("← Change Action", key="back_from_test_page"):
        st.session_state.workspace_action = None
        navigate_to_action_choice()
        st.rerun()
# Assuming this is placed before show_workspace_page()
def show_admin_portal():
    # ----------------------------------------------------
    # A. Access Control (CRITICAL)
    # ----------------------------------------------------
    admin_email = "admin@example.com"

    if st.session_state.get('logged_in_email') != admin_email:
        st.error("ACCESS DENIED: You must be logged in as the Administrator to view this portal.")
        
        if st.button("← Go to Login Page", key="admin_home_fail"):
            # If unauthorized, redirect to home/login page state
            navigate_to_login()
            st.rerun()
            
        return # Stop execution if unauthorized

    st.title("🛡️ System Administrator Portal")
    st.markdown("---")

    # --- CRITICAL VARIABLE DEFINITIONS (DATA FETCHING) ---
    metrics = get_system_metrics()
    user_df, workspace_df, model_df = get_all_management_data()
    
    # Fetch Activity and Annotations (These are safe local reads)
    activity_df = pd.read_sql("SELECT user_email, event_type, timestamp FROM activity_log ORDER BY timestamp DESC", get_db_connection_local())
    annotations_df = pd.read_sql("SELECT user_email, workspace_name, sentence, intent, entities_json, remarks, last_modified FROM annotations ORDER BY last_modified DESC", get_db_connection_local())

    # Calculate Best Model Row (Needed for Dashboard Row 2)
    best_model_row = None
    if not model_df.empty and 'accuracy' in model_df.columns:
        best_model_row = model_df.loc[model_df['accuracy'].idxmax()]

    # --- 1. SYSTEM HEALTH DASHBOARD (Ratio & Comparison Focus) ---
    st.header("1. System Performance Overview 📊")
    st.info("Visualizing efficiency, growth, and core performance trends.")

    MAX_USERS_GOAL = 50 
    MAX_ANNOTATIONS_GOAL = 1000

    # -----------------------------------------------------------
    # Row 1: Core Activity Trends (Line Chart & Comparative Bar)
    # -----------------------------------------------------------
    col_chart, col_comparison = st.columns(2)
    
    with col_chart:
        st.markdown("##### 📈 Daily User Activity Trend (Logins)")
        
        # Prepare data for Line Chart (requires datetime conversion)
        daily_logins = activity_df[activity_df['event_type'] == 'LOGIN'].copy()
        if not daily_logins.empty:
            daily_logins['date'] = pd.to_datetime(daily_logins['timestamp']).dt.date
            daily_counts = daily_logins.groupby('date').size().reset_index(name='Logins')

            st.line_chart(daily_counts.set_index('date'), use_container_width=True)
        else:
            st.info("No login activity to display.")

    with col_comparison:
        st.markdown("##### 👥 Total Counts & Workload")
        
        # Prepare data for Comparative Bar Chart (Users vs Workspaces vs Models)
        user_vs_ws_df = pd.DataFrame({
            'Category': ['Total Users', 'Total Workspaces', 'Total Models'],
            'Count': [metrics['total_users'], metrics['total_workspaces'], metrics['total_models']]
        })
        
        st.bar_chart(user_vs_ws_df.set_index('Category'))

    st.markdown("---")

    # -----------------------------------------------------------
    # Row 2: Performance & Data Quality Scorecard
    # -----------------------------------------------------------
    st.subheader("Performance Scorecard & Data Health")
    
    col_progress, col_score_2, col_score_3, col_score_4 = st.columns(4)
    
    # Calculate Annotation Progress
    annotation_progress_percent = min(100, (metrics['total_annotations'] / MAX_ANNOTATIONS_GOAL) * 100)

    # 1. Annotation Progress (Pie Chart)
    with col_progress:
        st.markdown("##### Data Completion Status")
        st.metric(label="Total Samples", value=metrics['total_annotations'])
        
        # Prepare data for Pie Chart
        annotations_done = metrics['total_annotations']
        annotations_remaining = max(0, MAX_ANNOTATIONS_GOAL - annotations_done)

        annotation_pie_df = pd.DataFrame({
            'Status': ['Annotated', 'Remaining'],
            'Count': [annotations_done, annotations_remaining]
        })
        
        try:
            import plotly.express as px
            fig = px.pie(
                annotation_pie_df, 
                values='Count', 
                names='Status', 
                height=180,
                hole=.6, # Creates a donut chart
                color_discrete_sequence=["#00CC96", "#EF553B"] # Green for done, Red for remaining
            )
            fig.update_traces(textposition='inside', textinfo='percent')
            fig.update_layout(margin=dict(t=10, b=0, l=0, r=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.warning("Install 'plotly' for Pie Chart viz.")


    # 2. Max Accuracy & F1 Score
    with col_score_2:
        st.markdown("##### Max Accuracy")
        if best_model_row is not None:
            st.metric("Top Score", f"{best_model_row['accuracy'] * 100:.2f}%")
            st.caption(f"Version: {best_model_row['version']}")
        else:
            st.info("No Models")
            
    with col_score_3:
        st.markdown("##### Model Reliability")
        if best_model_row is not None:
            st.metric("F1 Score", f"{best_model_row['f1_score'] * 100:.2f}%")
            st.caption(f"Samples: {best_model_row['trained_samples']}")
        else:
            st.info("N/A")

    # 4. Data Depth
    with col_score_4:
        avg_samples_per_model = model_df['trained_samples'].mean() if not model_df.empty and 'trained_samples' in model_df.columns else 0
        
        st.markdown("##### Avg. Training Depth")
        st.metric("Samples / Version", f"{avg_samples_per_model:.0f}")
        st.caption(f"Total versions: {metrics['total_models']}")


    st.markdown("---")
    
    # --- 2. MANAGEMENT TABLES ---
    st.header("2. Management Tables")
    tab_user, tab_workspace, tab_model, tab_activity, tab_feedback = st.tabs(["Users & Auth", "Workspaces", "Model History", "Activity Log & Metrics", "Feedback Log"])

    # --- 2A. USER MANAGEMENT (Tab - In-Line Delete) ---
    with tab_user:
        st.subheader("Users Management")
        st.caption("List of registered users. Click '❌' to delete with confirmation (cascading).")
        
        required_cols = ['name', 'email']
        
        if all(col in user_df.columns for col in required_cols) and not user_df.empty:
            
            st.markdown("##### Registered Users:")
            
            # Header
            col_name_h, col_email_h, col_action_h = st.columns([2, 3, 0.8])
            with col_name_h:
                st.markdown("**Name**")
            with col_email_h:
                st.markdown("**Email**")
            with col_action_h:
                st.markdown("**Action**")
            st.markdown("---") 

            # Rows with Delete Button
            for index, row in user_df.iterrows():
                user_name = row['name']
                user_email_to_delete = row['email']
                admin_email = "admin@example.com"
                is_admin = user_email_to_delete == admin_email
                is_current_user = user_email_to_delete == st.session_state.get('logged_in_email')
                
                col_name, col_email, col_action = st.columns([2, 3, 0.8])
                
                with col_name:
                    st.text(user_name)
                with col_email:
                    st.text(user_email_to_delete)
                
                with col_action:
                    delete_button_key = f"delete_user_{user_email_to_delete.replace('@', '_').replace('.', '_')}"
                    
                    if is_admin or is_current_user:
                        st.markdown(f"<p style='color:red; font-size:12px;'>🚫 Admin/Self</p>", unsafe_allow_html=True)
                    elif st.session_state.get(f'confirm_{delete_button_key}', False):
                        st.warning("Confirm delete?")
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            st.button("✅ Yes", 
                                key=f"confirm_yes_{delete_button_key}", 
                                type="primary",
                                on_click=delete_user_data,
                                args=(user_email_to_delete,),
                            )
                        with col_no:
                            st.button("❌ No", 
                                key=f"confirm_no_{delete_button_key}", 
                                on_click=lambda k=delete_button_key: st.session_state.pop(f'confirm_{k}', None)
                            )
                    else:
                        st.button(
                            "❌", 
                            key=delete_button_key, 
                            type="secondary", 
                            help=f"Delete {user_email_to_delete} and all associated data.",
                            on_click=lambda k=delete_button_key: st.session_state.update({f'confirm_{k}': True})
                        )

            st.markdown("---")
        
        else:
            st.info("No registered users found (apart from the root admin).")
            
    # --- 2B. WORKSPACES MANAGEMENT (Tab - In-Line Delete/Download) ---
    with tab_workspace:
        st.subheader("Workspaces Management")
        st.caption("All workspaces in the system. Delete is cascading.")
        
        required_cols = ['user_email', 'workspace_name', 'domain', 'last_modified']
        
        if all(col in workspace_df.columns for col in required_cols) and not workspace_df.empty:
            
            # Header
            col_email_h, col_ws_h, col_domain_h, col_modified_h, col_download_h, col_action_h = st.columns([2, 1.5, 1, 1.5, 1, 0.8])
            with col_email_h:
                st.markdown("**User Email**")
            with col_ws_h:
                st.markdown("**Workspace Name**")
            with col_domain_h:
                st.markdown("**Domain**")
            with col_modified_h:
                st.markdown("**Last Mod.**")
            with col_download_h:
                st.markdown("**Download**")
            with col_action_h:
                st.markdown("**Action**")
            st.markdown("---") 

            # Rows with Download and Delete Buttons
            for index, row in workspace_df.iterrows():
                ws_email = row['user_email']
                ws_name = row['workspace_name']
                
                col_email, col_ws, col_domain, col_modified, col_download, col_action = st.columns([2, 1.5, 1, 1.5, 1, 0.8])
                
                with col_email:
                    st.text(ws_email)
                with col_ws:
                    st.text(ws_name)
                with col_domain:
                    st.text(row['domain'])
                with col_modified:
                    st.text(str(row['last_modified'])[:10])
                
                # --- DOWNLOAD BUTTON LOGIC ---
                with col_download:
                    filename, data_bytes = fetch_dataset_blob_for_admin(ws_email, ws_name) 
                    
                    if data_bytes:
                        st.download_button(
                            label="⬇️ CSV",
                            data=data_bytes,
                            file_name=filename,
                            mime="text/csv",
                            key=f"download_ws_{ws_name}_{ws_email}",
                            use_container_width=True
                        )
                    else:
                        st.markdown("<p style='font-size: 12px; color: #aaa;'>No Dataset</p>", unsafe_allow_html=True)

                # --- DELETE BUTTON LOGIC ---
                with col_action:
                    delete_button_key = f"delete_ws_{ws_name.replace(' ', '_')}_{ws_email.replace('@', '_').replace('.', '_')}"
                    
                    if st.session_state.get(f'confirm_{delete_button_key}', False):
                        st.warning("Confirm delete?")
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            st.button("✅ Yes", 
                                key=f"confirm_yes_{delete_button_key}", 
                                type="primary",
                                on_click=delete_workspace_data, 
                                args=(ws_email, ws_name),
                            )
                        with col_no:
                            st.button("❌ No", 
                                key=f"confirm_no_{delete_button_key}", 
                                on_click=lambda k=delete_button_key: st.session_state.pop(f'confirm_{k}', None)
                            )
                    else:
                        st.button(
                            "❌", 
                            key=delete_button_key, 
                            type="secondary", 
                            help=f"Delete workspace {ws_name} and all data.",
                            on_click=lambda k=delete_button_key: st.session_state.update({f'confirm_{k}': True})
                        )
            st.markdown("---")
        else:
            st.info("No workspaces found.")


    # --- 2C. MODEL MANAGEMENT (Tab - In-Line Delete) ---
    with tab_model:
        st.subheader("Model Version Management")
        st.caption("All trained model versions in the system.")
        
        # Merge model_df with workspace_df to fetch the Domain
        model_display_df = model_df.merge(
            workspace_df[['user_email', 'workspace_name', 'domain']], 
            on=['user_email', 'workspace_name'], 
            how='left'
        )
        
        required_cols = ['version', 'workspace_name', 'domain', 'user_email', 'train_date', 'accuracy', 'f1_score']
        
        if all(col in model_display_df.columns for col in required_cols) and not model_display_df.empty:
            
            # Sort by date for cleaner display
            model_df_sorted = model_display_df.sort_values(by='train_date', ascending=False)
            
            # Header
            col_v_h, col_ws_h, col_domain_h, col_email_h, col_date_h, col_acc_h, col_f1_h, col_action_h = st.columns([1.5, 1.5, 1, 2, 1.5, 1, 1, 0.8])
            
            with col_v_h:
                st.markdown("**Version**")
            with col_ws_h:
                st.markdown("**Workspace**")
            with col_domain_h:
                st.markdown("**Domain**")
            with col_email_h:
                st.markdown("**Email**")
            with col_date_h:
                st.markdown("**Train Date**")
            with col_acc_h:
                st.markdown("**Acc.**")
            with col_f1_h:
                st.markdown("**F1 Score**")
            with col_action_h:
                st.markdown("**Action**")
            st.markdown("---") 

            # Rows with Delete Button
            for index, row in model_df_sorted.iterrows():
                model_version = row['version']
                ws_name = row['workspace_name']
                user_email = row['user_email']

                unique_identifier = f"{model_version}_{ws_name}_{user_email}".replace(' ', '_').replace('@', '_').replace('.', '_')
                delete_button_key = f"delete_model_{unique_identifier}"
                
                col_v, col_ws, col_domain, col_email, col_date, col_acc, col_f1, col_action = st.columns([1.5, 1.5, 1, 2, 1.5, 1, 1, 0.8])
                
                with col_v:
                    st.text(model_version)
                with col_ws:
                    st.text(ws_name)
                with col_domain:
                    st.text(row['domain']) 
                with col_email:
                    st.text(user_email)
                with col_date:
                    st.text(str(row['train_date'])[:10]) 
                with col_acc:
                    st.text(f"{row['accuracy']:.2f}") 
                with col_f1:
                    st.text(f"{row['f1_score']:.2f}") 
                
                with col_action:
                    
                    if st.session_state.get(f'confirm_{delete_button_key}', False):
                        st.warning("Confirm delete?")
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            st.button("✅ Yes", 
                                key=f"confirm_yes_{delete_button_key}", 
                                type="primary",
                                on_click=delete_model_version, 
                                args=(model_version,),
                            )
                        with col_no:
                            st.button("❌ No", 
                                key=f"confirm_no_{delete_button_key}", 
                                on_click=lambda k=delete_button_key: st.session_state.pop(f'confirm_{k}', None)
                            )
                    else:
                        st.button(
                            "❌", 
                            key=delete_button_key, 
                            type="secondary", 
                            help=f"Delete model version {model_version}.",
                            on_click=lambda k=delete_button_key: st.session_state.update({f'confirm_{k}': True})
                        )
            st.markdown("---")
        else:
            st.info("No model versions found.")
            
    # --- 2D. ACTIVITY LOG & TIME SPENT (NEW TAB) ---
    with tab_activity:
        st.subheader("User Activity Log")
        st.caption("Raw login and logout events.")
        
        # Check if activity_df is empty before processing
        if not activity_df.empty:
            st.dataframe(activity_df, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Time Spent Analysis (Simplified)")
            st.info("This is a simplified analysis assuming time between LOGIN and LOGOUT is the session duration.")
            
            # Convert timestamp to datetime objects
            activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])
            
            # --- Calculate Session Duration ---
            session_data = []
            
            for email in activity_df['user_email'].unique():
                # Get events for the user, sorted by time
                user_events = activity_df[activity_df['user_email'] == email].sort_values('timestamp')
                
                login_time = None
                for index, row in user_events.iterrows():
                    if row['event_type'] == 'LOGIN':
                        login_time = row['timestamp']
                    elif row['event_type'] == 'LOGOUT' and login_time is not None:
                        session_duration = row['timestamp'] - login_time
                        session_data.append({
                            'user_email': email,
                            'login_time': login_time,
                            'logout_time': row['timestamp'],
                            'duration': session_duration
                        })
                        login_time = None # Reset for the next session
                        
            if session_data:
                session_df = pd.DataFrame(session_data)
                
                # Calculate Total Time and Average Time
                total_duration = session_df.groupby('user_email')['duration'].sum()
                avg_duration = session_df.groupby('user_email')['duration'].mean()
                
                metrics_df = pd.DataFrame({
                    'Total Time Spent': total_duration.dt.total_seconds().apply(lambda x: f"{x/3600:.2f} hrs"),
                    'Avg Session Time': avg_duration.dt.total_seconds().apply(lambda x: f"{x/60:.2f} mins")
                }).reset_index()
                
                st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            else:
                st.info("Not enough paired LOGIN/LOGOUT events to calculate time spent.")
        else:
            st.warning("No user activity logs found yet.")

    # ----------------------------------------------------
    # 2E. FEEDBACK LOG (STABILIZED AND FUNCTIONAL TAB)
    # ----------------------------------------------------
    with tab_feedback:
        st.subheader("📝 Feedback Log & Data Review")
        st.caption("Review user-submitted corrections and general system feedback.")

        if annotations_df.empty:
            st.info("No annotations or feedback entries found in the database yet.")
        else:
            
            # Initialize Selector State (for stability)
            if "feedback_log_view_selector" not in st.session_state:
                 st.session_state["feedback_log_view_selector"] = "All Feedback (NLU + General)"
            
            # Filter DataFrames upfront
            general_feedback = annotations_df[annotations_df['intent'] == 'general_system_feedback']
            nlu_corrections = annotations_df[annotations_df['intent'] != 'general_system_feedback']
            
            # --- START FUNCTIONAL FILTERING FORM ---
            # Now includes a Submit button to apply the filter change
            with st.form("feedback_filter_form_functional"):
                st.markdown("##### Filter Options")
                
                col_info, col_filter = st.columns([1, 1])
                
                with col_info:
                    st.info(f"Found **{len(general_feedback)}** General System Feedback items.")
                with col_filter:
                    # The selectbox value is bound to the key
                    filter_option = st.selectbox(
                        "Select Data View:",
                        options=["All Feedback (NLU + General)", "NLU Corrections Only", "General System Feedback Only"],
                        key="feedback_log_view_selector"
                    )
                
                # CRITICAL: Add a functional submit button. Clicking this is the *only* way to trigger the filter.
                submitted = st.form_submit_button("Apply Filter", type="primary")

            # --- FILTER APPLICATION LOGIC ---
            
            # The filtering logic runs on every rerun. 
            # When 'submitted' is True, the selectbox value in st.session_state has been updated.
            
            current_filter_choice = st.session_state.get("feedback_log_view_selector")
            
            if current_filter_choice == "NLU Corrections Only":
                display_df = nlu_corrections
            elif current_filter_choice == "General System Feedback Only":
                display_df = general_feedback
            else:
                display_df = annotations_df
            
            st.markdown("---")
            st.markdown(f"##### Showing {len(display_df)} Entries (Filtered by: {current_filter_choice})")
            
            # Select columns to display for clarity
            display_cols = ['last_modified', 'user_email', 'workspace_name', 'intent', 'sentence', 'remarks']
            
            # Apply styling to highlight General System Feedback rows
            def highlight_general_feedback(s):
                is_general = s['intent'] == 'general_system_feedback'
                return ['background-color: #331a1a' if is_general else '' for _ in s]

            st.dataframe(
                display_df[display_cols].style.apply(highlight_general_feedback, axis=1),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "last_modified": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
                    "intent": st.column_config.TextColumn("Intent/Type"),
                    "remarks": st.column_config.TextColumn("Remarks/Feedback", width="large")
                }
            )
#=====================================
#Feedback
#=====================================

def show_feedback_page():
    user_email = st.session_state.get('logged_in_email')
    workspace_name = st.session_state.get('current_workspace')
    domain = st.session_state.get('current_domain')
    
    # Safety check
    if not workspace_name:
        st.error("Please select a workspace before using the Feedback Module.")
        if st.button("Go to Home"): navigate_to_home(); st.rerun()
        return

    st.markdown(f"<div class='title'>📝 Feedback & Data Logger: <span style='color:#6EC6FF;'>{workspace_name}</span></div>", unsafe_allow_html=True)
    st.info("Log NLU corrections OR general system feedback.")

    # --------------------------------------------------------------------------
    # --- CRITICAL FIX: LOGIC RUNS BEFORE FORM DEFINITION (No Callback Needed) ---
    # --------------------------------------------------------------------------
    
    # Ensure state variables are initialized (This avoids errors if the user re-runs the page)
    if 'feedback_sentence_input' not in st.session_state: st.session_state.feedback_sentence_input = ""
    if 'feedback_remarks_input' not in st.session_state: st.session_state.feedback_remarks_input = ""
    if 'is_general_feedback_checkbox' not in st.session_state: st.session_state.is_general_feedback_checkbox = False

    # FIX: Copy query to remarks *before* form definition, based on current state values
    if st.session_state.is_general_feedback_checkbox:
        # If the checkbox is currently checked...
        if st.session_state.feedback_sentence_input and st.session_state.feedback_sentence_input.strip():
            # Only copy if the remarks field is empty or was set by a previous similar copy (startswith check)
            if not st.session_state.feedback_remarks_input.strip() or st.session_state.feedback_remarks_input.startswith("User Query:"):
                 st.session_state.feedback_remarks_input = f"User Query: {st.session_state.feedback_sentence_input}"


    # --------------------------------------------------------------------------
    
    
    # 1. Input and Prediction Form
    with st.form(key='standalone_feedback_form', clear_on_submit=True):
        
        # --- GENERAL FEEDBACK CHECKBOX (CRITICAL: Removed on_change) ---
        is_general_feedback = st.checkbox(
            "Is this General System Feedback? (Check this to bypass Intent/Entity requirements)",
            key="is_general_feedback_checkbox"
        )

        # --- INPUT FIELD ---
        sentence_input = st.text_input(
            "Enter or Paste User Query/Feedback:",
            key="feedback_sentence_input",
            placeholder="e.g., The bot misunderstood my flight dates. OR The website theme is too dark."
        )

        st.markdown("---")
        st.markdown("### Model Prediction & Correction")
        
        predicted_intent = "Awaiting input..."
        current_entities_str = ""
        entities_dict = {}

        # --- RUN LIVE PREDICTION ---
        if sentence_input.strip() and not is_general_feedback:
            # Run prediction only if there's text AND it's for NLU correction
            intent, entities_json = predict_intent_and_entities(sentence_input.strip(), domain)
            
            try:
                entities_dict = json.loads(entities_json)
            except json.JSONDecodeError:
                entities_dict = {"Error": "Invalid JSON format"}
                
            predicted_intent = intent
            current_entities_str = ", ".join(f"{k}:{v}" for k, v in entities_dict.items())

        # --- DISPLAY PREDICTIONS ---
        
        # Predicted Intent (Read-only, for context)
        st.text_input(
            "Predicted Intent:",
            value=predicted_intent,
            disabled=True,
            key="feedback_predicted_intent_display"
        )
        
        # Correction Fields
        domain_intents = DOMAINS.get(domain, {}).get("intents", [])
        
        col_corr_intent, col_remarks = st.columns([1, 1])
        
        with col_corr_intent:
            correct_intent = st.selectbox(
                "Correct Intent (Select/Type):",
                options=['-- Select --'] + domain_intents + ([predicted_intent] if predicted_intent not in domain_intents and predicted_intent != "Awaiting input..." else []) + ["default_fallback"],
                key="feedback_correct_intent_select",
                disabled=is_general_feedback # Disable if general feedback is checked
            )
        
        with col_remarks:
            # Value is set by the pre-processing logic above via st.session_state.feedback_remarks_input
            feedback_remarks = st.text_area(
                "Feedback Remarks (Optional):",
                key="feedback_remarks_input", 
                placeholder="e.g., The prediction was incorrect because... OR The website navigation is confusing."
            )
            
        edited_entities_str = st.text_area(
            "Entities (Editable - Format: key:value, key:value):",
            value=current_entities_str,
            key="feedback_entities_editable",
            disabled=is_general_feedback # Disable if general feedback is checked
        )

        # 2. Submission Button
        save_button = st.form_submit_button(
            label='✅ Save Feedback', 
            type="primary",
            use_container_width=True
        )

        if save_button:
            # --- Validation ---
            if not st.session_state.feedback_sentence_input.strip():
                st.error("Please enter a User Query/Feedback before saving.")
                return
            
            final_intent = st.session_state.feedback_correct_intent_select
            entities_json_to_save = "{}"
            feedback_text = st.session_state.feedback_remarks_input.strip()
            
            if is_general_feedback:
                # --- GENERAL FEEDBACK PATH ---
                final_intent = "general_system_feedback"
                if not feedback_text:
                    # NOTE: This check should now only fail if the user manually cleared the auto-copied text
                    st.error("Please provide remarks for general system feedback.")
                    return
            
            else:
                # --- NLU CORRECTION PATH ---
                if not final_intent or final_intent == '-- Select --':
                    st.error("Please select or type the Correct Intent for NLU correction.")
                    return
                
                # --- Entity Parsing for Saving ---
                entities_dict_to_save = {}
                try:
                    if st.session_state.feedback_entities_editable.strip():
                        parts = st.session_state.feedback_entities_editable.split(',')
                        for part in parts:
                            if ':' in part:
                                k, v = part.split(':', 1)
                                entities_dict_to_save[k.strip()] = v.strip()
                    entities_json_to_save = json.dumps(entities_dict_to_save)
                        
                except Exception:
                    st.error("Error parsing edited entities. Please ensure format is: key:value, key:value")
                    return
            
            # --- SAVE TO DB ---
            if save_annotation_to_db(
                workspace_name, 
                user_email, 
                st.session_state.feedback_sentence_input.strip(), 
                final_intent, # Will be corrected intent or "general_system_feedback"
                entities_json_to_save, 
                feedback_text # Will contain remarks for both paths
            ):
                st.success(f"Feedback Saved! Logged under Intent: {final_intent}")
                st.toast("Feedback saved successfully! Data ready for review.", icon="📝")
                
                # Rerun to clear the form (handled by clear_on_submit=True)
                st.rerun()

    # --- Back Button ---
    st.markdown("---")
    if st.button("← Change Action", key="back_from_feedback"):
        st.session_state.workspace_action = None
        navigate_to_action_choice()
        st.rerun()
# ===============================
# PAGE DISPLAY FUNCTIONS
# ===============================

def show_training_page():
    user_email = st.session_state.get('logged_in_email')
    workspace_name = st.session_state.get('current_workspace')
    domain = st.session_state.get('current_domain')
    
    # Safety check
    if not user_email or not workspace_name:
        st.error("Workspace or user information missing. Please navigate back to Home.")
        return 
        
    domain_display = DOMAINS.get(domain, {}).get("icon", "❓") + " " + domain
    st.subheader(f"📚 Upload & Train - {domain_display} Domain")

    # --- 1. Data Upload Section ---
    st.subheader("1. Upload Dataset (CSV)")
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing your example sentences. Recommended columns: 'User Input', 'Intent', 'Entities'.", 
        type=['csv'],
        key="csv_uploader_train_page"
    )
    
    # Check if a dataset already exists in the DB
    local_cursor_check = conn.cursor()
    local_cursor_check.execute(
        "SELECT filename FROM datasets WHERE user_email=? AND workspace_name=?", 
        (user_email, workspace_name)
    )
    existing_dataset = local_cursor_check.fetchone()
    dataset_is_saved = existing_dataset is not None
    
    if dataset_is_saved:
        st.success(f"Dataset **{existing_dataset[0]}** is currently saved for this workspace.")
    
    if uploaded_file is not None:
        # Display a preview
        try:
            # Need to reset the file pointer before reading to dataframe 
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            st.info(f"File: {uploaded_file.name} | Rows: {len(df)}")
            
            # Save button within the upload context
            if st.button(f"Save '{uploaded_file.name}' to Workspace", key="save_dataset_btn", type="primary"):
                # Read file data as bytes (BLOB)
                uploaded_file.seek(0) # Reset file pointer again
                file_data_bytes = uploaded_file.getvalue()
                
                local_cursor_train = conn.cursor()
                
                # Use UPSERT (INSERT OR REPLACE) logic on datasets table
                local_cursor_train.execute(
                    """
                    INSERT INTO datasets (user_email, workspace_name, filename, data) 
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_email, workspace_name) DO UPDATE SET 
                    filename = excluded.filename, 
                    data = excluded.data
                    """,
                    (user_email, workspace_name, uploaded_file.name, Binary(file_data_bytes))
                )
                
                # Update the last_modified timestamp here to bust the cache in the Annotation page
                local_cursor_train.execute("UPDATE workspaces SET last_modified=CURRENT_TIMESTAMP WHERE user_email=? AND workspace_name=?", (user_email, workspace_name))
                conn.commit()
                
                # --------------------------------------------------------
                # <<<<<<<<<<<<<<<< START NEW LOGIC BLOCK (ROBUST) >>>>>>>>>>>>
                # --------------------------------------------------------

                # 1. Read the saved CSV back into a DataFrame (using the already uploaded file object)
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file) 

                # --- CRITICAL DEBUG STEP ---
                # This line shows you exactly what headers the code is seeing
                st.info(f"DEBUG: Found these columns in the CSV: {df.columns.tolist()}")
                # --- END DEBUG STEP ---


                # 2. Identify the Sentence, Intent, and Entity columns (case-insensitive search)
                sentence_col = next((col for col in df.columns if col.lower() in ['text', 'sentence', 'utterance', 'input', 'user_statement', 'user input', 'user_query']), None)
                intent_col = next((col for col in df.columns if col.lower() in ['intent', 'intent_name', 'label', 'class']), None) 
                entity_col = next((col for col in df.columns if col.lower() in ['entities', 'entity', 'entities_json', 'slots', 'data']), None)


                if sentence_col and intent_col and entity_col:
                    # 3. Prepare data for batch insertion (drop NaNs in key columns)
                    import_df = df[[sentence_col, intent_col, entity_col]].dropna()
                    import_df = import_df.rename(columns={
                        sentence_col: 'sentence', 
                        intent_col: 'intent', 
                        entity_col: 'entity_input'
                    })
                    
                    count = 0
                    # 4. Process each row and insert into the annotations table
                    for index, row in import_df.iterrows():
                        try:
                            sentence = row['sentence'].strip()
                            intent = row['intent'].strip()
                            entity_input_string = row['entity_input']
                            
                            # Use existing utility functions to convert entity string format to DB JSON format
                            entities_list = entities_input_to_dict(entity_input_string) 
                            entities_json = dict_to_entities_json_string(entities_list) 
                            
                            # Save the record (UPSERT)
                            if save_annotation_to_db(workspace_name, user_email, sentence, intent, entities_json):
                                count += 1
                        except Exception:
                            # Skip rows with invalid data
                            continue
                            
                    if count > 0:
                        st.success(f"✅ Success! Found and imported **{count}** pre-labeled examples from CSV into annotations.")
                    else:
                        st.warning("Found label columns but imported 0 records. Check for empty rows.")
                else:
                    # Debugging instructions if automatic detection still fails
                    st.error("AUTOMATED IMPORT FAILED: Please check the 'DEBUG' message above for the exact column names in your CSV. You may need to edit the search lists manually.")
                    st.warning("CSV saved. No recognizable intent or entity columns found. You must manually annotate the data.")
                
                # --------------------------------------------------------
                # <<<<<<<<<<<<<<<<< END NEW LOGIC BLOCK >>>>>>>>>>>>>>>>>>
                # --------------------------------------------------------

                # Invalidate sentence cache and reset index if new data is uploaded/saved
                st.session_state.sentences_df = None
                st.session_state.annotation_index = 0
                
                st.success(f"✅ Data for **{workspace_name}** saved.")
                time.sleep(1) # Delay for visual confirmation
                st.rerun() # Rerun to refresh the success message and clear the file uploader

        except Exception as e:
            st.error(f"Error processing or saving dataset: {e}")

    # --- 2. Train Model Section ---
    st.subheader("2. Train NLU Model")
    
    # Check for annotated data before allowing training
    annotated_data = pd.read_sql("SELECT * FROM annotations WHERE user_email=? AND workspace_name=?", conn, params=(user_email, workspace_name))
    annotation_count = len(annotated_data)

    if dataset_is_saved:
        if annotation_count > 0:
            
            # Only show the form if minimum 10 samples are annotated
            if annotation_count >= 10:
                st.success(f"✅ Ready to Train! Found **{annotation_count}** annotated samples.")
            
                # --- MODEL SELECTION AND TRAINING FORM (SINGLE COLUMN) ---
                with st.form("training_form"):
                    # Determine next version based on history (assumes get_model_history exists)
                    try:
                        history_df = get_model_history(user_email, workspace_name)
                    except NameError:
                        st.error("Error: The 'get_model_history' function is missing. Please define it.")
                        # Do not return, allow the user to see the error and fix it
                        history_df = pd.DataFrame() # Fallback to empty DF
                        
                    current_versions = len(history_df)
                    default_version = f"v{current_versions + 1}"
                    
                    # 1. Model Version Name
                    new_version = st.text_input(
                        "Model Version Name", 
                        value=default_version, 
                        key="train_model_version"
                    )
                    
                    # 2. Model Architecture Selectbox
                    model_choice = st.selectbox(
                        "Choose NLU Model Architecture",
                        options=[
                            "Transformer (BERT/RoBERTa) - High Accuracy/Slow",
                            "Simple Classifier (spaCy-like) - Fast/Lower Accuracy"
                        ],
                        key="model_select",
                        help="Choose a model based on your accuracy and speed requirements."
                    )
                    # Convert choice to the internal key for simulation
                    model_name = "trf_model" if "transformer" in model_choice.lower() else "simple_model"
                    
                    # 3. Epochs Input
                    epochs = st.number_input(
                        "Epochs", 
                        min_value=10, 
                        max_value=200, 
                        value=50, 
                        step=10, 
                        key="epochs_input"
                    )
                    notes = st.text_area("Training Notes", placeholder="e.g., Initial run with default settings", key="train_notes")
                    
                    submitted = st.form_submit_button("🚀 Start Model Training", type="primary", use_container_width=True)

                    if submitted:
                        if new_version.strip():
                            # Call the core simulation function
                            acc, f1, samples = _simulate_nlu_training_core( 
                                user_email, 
                                workspace_name, 
                                new_version.strip(), 
                                model_name, 
                                epochs, 
                                notes
                            )
                            
                            # CRITICAL: Check the return values
                            if acc and f1:
                                st.success(f"✅ Training Complete for **{new_version}**!")
                                st.info(f"Model: {model_choice} | Samples: {samples} | Epochs: {epochs}")
                                
                                col_acc, col_f1 = st.columns(2)
                                col_acc.metric(label="Accuracy", value=f"{acc*100:.2f}%")
                                col_f1.metric(label="F1 Score", value=f"{f1*100:.2f}%")
                                
                                time.sleep(1) 
                                st.rerun() # Rerunning ensures the success message persists
                            elif samples is not None and samples < 10:
                                # This block should theoretically not be hit due to the check before the form
                                st.error(f"Cannot start training. You need at least 10 annotated samples. Found: {samples}")
                            else:
                                # This catches the IntegrityError/Version Conflict or any other save failure
                                st.error("Training failed to save. Did you try a new Model Version Name? (Check logs for details)")
                                
                        else:
                            st.error("Please provide a version name.")
                        
                # 2. Annotation Button (Visible if dataset is saved, even if training is possible)
                if st.button("Continue Annotation →", use_container_width=True, key="go_to_annotate_continue", type="secondary"):
                    set_workspace_action("Annotate")
                    st.rerun()
            else:
                # Dataset saved, but less than 10 annotations. Show the 'Go to Annotation' button prominently.
                st.warning(f"Dataset is ready! Please **Annotate Data** first to create labels for custom training. You need at least 10 samples. Found: **{annotation_count}**.")
                if st.button("Go to Annotation Page...", key="go_to_annotate_from_train", type="primary", use_container_width=True):
                    set_workspace_action("Annotate")
                    st.rerun()
        else:
            # Dataset saved, but no annotations. Show the 'Go to Annotation' button prominently.
            st.warning("Dataset is ready! Please **Annotate Data** first to create labels for custom training.")
            if st.button("Go to Annotation Page...", key="go_to_annotate_from_train", type="primary", use_container_width=True):
                set_workspace_action("Annotate")
                st.rerun()
    else:
        st.warning("Please upload a dataset (CSV) in section 1 before training.")

    # --- Back Buttons ---
    st.markdown("---")
    if st.button("← Change Action", key="back_from_train_page"):
        st.session_state.workspace_action = None
        navigate_to_action_choice()
        st.rerun()
# ==============================
# HOME PAGE
# ==============================
def get_workspaces_for_user(user_email):
    """Retrieves all workspaces associated with the logged-in user."""
    local_cursor = conn.cursor()
    local_cursor.execute(
        "SELECT workspace_name, domain FROM workspaces WHERE user_email=? ORDER BY last_modified DESC", 
        (user_email,)
    )
    return local_cursor.fetchall()

def get_annotated_data_count(user_email, workspace_name):
    """Retrieves the count of annotations for the current workspace."""
    # Uses the cached DB connection 'conn' defined globally
    local_cursor = conn.cursor()
    local_cursor.execute(
        "SELECT COUNT(*) FROM annotations WHERE user_email = ? AND workspace_name = ?",
        (user_email, workspace_name)
    )
    return local_cursor.fetchone()[0]

def show_home_page():
    if not st.session_state.logged_in_email:
        navigate_to_login()
        st.rerun()
        return

    user_email = st.session_state.logged_in_email
    st.markdown("<div class='title'>Welcome Back!</div>", unsafe_allow_html=True)
    st.subheader("Your BuddyBot Workspaces")

    workspaces = get_workspaces_for_user(user_email)
    num_workspaces = len(workspaces)
    
    # Initialize columns for layout: 3 cards per row
    cols = st.columns(3) 

    if workspaces:
        st.markdown(f"**You have {num_workspaces} existing projects. Select one to activate:**")
        st.markdown("---")
        
        # 1. Display existing workspaces
        for i, (name, domain) in enumerate(workspaces):
            # Select the column for the current card (0, 1, or 2)
            col_index = i % 3
            
            with cols[col_index]:
                # Custom Card Display
                icon = DOMAINS.get(domain, {}).get("icon", "❓")
                description = DOMAINS.get(domain, {}).get("description", "No domain description available.")
                
                is_current = st.session_state.current_workspace == name
                
                st.markdown(f"""
                    <div class="domain-card" style='height: 180px;'>
                        <h3>{icon} {name}</h3>
                        <p style='color: #6EC6FF; font-weight: bold;'>Domain: {domain}</p>
                        <p>{description}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.button(
                    "Active Bot" if is_current else "Activate Bot",
                    key=f"activate_ws_{name.replace(' ', '_')}",
                    use_container_width=True,
                    # Change type to 'secondary' if it is the current one
                    type="secondary" if is_current else "primary",
                    disabled=is_current,
                    on_click=activate_existing_workspace,
                    args=(name, domain)
                )
                st.markdown("<br>", unsafe_allow_html=True)

        # 2. Add the "Create New Project" button to the end of the last row or start of a new one
        # If the number of projects perfectly fills the row (e.g., 3, 6, 9), create a new row/columns
        if num_workspaces % 3 == 0:
            cols = st.columns(3) # Create a new row of columns

        # Calculate the index for the 'Create New Project' card
        create_card_index = num_workspaces % 3
        with cols[create_card_index]:
            # Use custom HTML/CSS to create a card-like button
            st.markdown(f"""
                <div class="domain-card" style='border: 2px dashed #6EC6FF; height: 180px; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
                    <h3 style='color: #6EC6FF; margin-bottom: 5px; font-size: 36px;'>➕</h3>
                    <p style='color: #6EC6FF; font-weight: bold; font-size: 16px;'>Create New Project</p>
                    <p style='font-size: 12px; color: #ccc;'>Start a new NLU bot</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Streamlit button to handle the click action, placed directly under the custom card
            if st.button(
                "Go to Project Creation", 
                use_container_width=True, 
                key="go_to_create_btn_card", 
                type="primary"
            ):
                navigate_to_create_workspace()
                st.rerun()
                
        st.markdown("---")

    else:
        st.markdown("### Create Your First BuddyBot Workspace")
        with st.container(border=True):
            st.markdown("""
                <p>It looks like you don't have any workspaces yet. Start by creating a new project 
                and selecting a domain to specialize your NLU model.</p>
            """, unsafe_allow_html=True)
            if st.button("Create New Workspace", key="go_to_create_btn_empty"):
                navigate_to_create_workspace()
                st.rerun()

# ==============================
# WORKSPACE CREATION PAGE
# ==============================
def show_create_workspace_page():
    st.markdown("<div class='title'>1. Name Your Workspace</div>", unsafe_allow_html=True)
    with st.form(key="workspace_name_form"):
        workspace_name_input = st.text_input(
            "Enter a name for your new Bot/Workspace (e.g., 'Finance-QBot', 'Sports-Stats')",
            value=st.session_state.temp_workspace_name
        )
        st.session_state.temp_workspace_name = workspace_name_input
        submitted = st.form_submit_button("Next: Choose Domain", type="primary")

    if submitted and workspace_name_input:
        st.markdown("<div class='title'>2. Choose Your Domain</div>", unsafe_allow_html=True)
        st.markdown("Select the area your NLU bot will specialize in:")

        # Domain selection layout: 3 cards per row
        domains_list = list(DOMAINS.keys())
        num_domains = len(domains_list)
        
        # Calculate rows and use st.columns dynamically
        for i in range(0, num_domains, 3):
            cols = st.columns(3)
            for j in range(3):
                domain_index = i + j
                if domain_index < num_domains:
                    domain_name = domains_list[domain_index]
                    domain_data = DOMAINS[domain_name]
                    
                    with cols[j]:
                        st.markdown(f"""
                            <div class="domain-card">
                                <h3>{domain_data['icon']} {domain_name}</h3>
                                <p>{domain_data['description']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.button(
                            f"Select {domain_name}",
                            key=f"select_domain_{domain_name.replace(' ', '_')}",
                            use_container_width=True,
                            type="primary",
                            on_click=finalize_workspace_creation,
                            args=(workspace_name_input, domain_name)
                        )
                        st.markdown("<br>", unsafe_allow_html=True)
                        
    if st.button("← Back to Workspaces Home", key="back_from_create_workspace"):
        navigate_to_home()
        st.rerun()


# ==============================
# WORKSPACE ACTION CHOICE PAGE (MODIFIED FOR 4 COLUMNS)
# ==============================
def show_action_choice_page():
    if not st.session_state.logged_in_email or not st.session_state.current_workspace:
        navigate_to_home()
        st.rerun()
        return

    workspace_name = st.session_state.current_workspace
    domain = st.session_state.current_domain
    icon = DOMAINS.get(domain, {}).get("icon", "❓")
    
    st.markdown(f"<div class='title'>🛠️ Action Center: <span style='color:#6EC6FF;'>{icon} {workspace_name}</span></div>", unsafe_allow_html=True)
    st.subheader(f"What would you like to do in the **{domain}** domain?")
    
    # Allocating 6 columns now to fit the new action, or 5 columns if you prefer a tighter layout.
    # We will use 5 columns and group Feedback with Annotation/Test tools.
    col1, col2, col3, col4, col5, col6 = st.columns(6) 

    # Col 1: ANNOTATE 
    with col1:
        st.markdown(
            """
            <div class='domain-card' style='height: 150px;'>
                <h3>🏷️ Annotate Data</h3>
                <p>Label raw text with Intents and Entities.</p>
            </div>
            """, unsafe_allow_html=True
        )
        st.button(
            "Start Annotation", 
            key="action_btn_annotate",
            use_container_width=True, 
            on_click=set_workspace_action, 
            args=("Annotate",)
        )

    # Col 2: Train
    with col2:
        st.markdown(
            """
            <div class='domain-card' style='height: 150px;'>
                <h3>🧠 Train Model</h3>
                <p>Build an NLU model from annotated data.</p>
            </div>
            """, unsafe_allow_html=True
        )
        st.button(
            "Upload & Train Bot", 
            key="action_btn_train",
            use_container_width=True, 
            on_click=set_workspace_action, 
            args=("Train",)
        )

    # Col 3: Test
    with col3:
        st.markdown(
            """
            <div class='domain-card' style='height: 150px;'>
                <h3>💬 Test & Chat</h3>
                <p>Interactively test the latest trained model.</p>
            </div>
            """, unsafe_allow_html=True
        )
        st.button(
            "Start Chatting", 
            key="action_btn_test",
            use_container_width=True, 
            on_click=set_workspace_action, 
            args=("Test",)
        )
        
    # Col 4: Feedback Module (NEW ACTION)
    with col4:
        st.markdown(
            """
            <div class='domain-card' style='height: 150px;'>
                <h3>📝 Feedback Review</h3>
                <p>Review and process saved error reports.</p>
            </div>
            """, unsafe_allow_html=True
        )
        # Inside show_action_choice_page, for the Feedback Module card:

        st.button(
            "Go to Feedback", 
            key="action_btn_feedback",
            use_container_width=True, 
    on_click=set_workspace_action, 
    args=("Feedback",) # Use a new action state: "Feedback"
)

    # Col 5: Evaluate
    with col5:
        st.markdown(
            """
            <div class='domain-card' style='height: 150px;'>
                <h3>📊 Evaluate Models</h3>
                <p>Run versioned evaluations and check performance metrics.</p>
            </div>
            """, unsafe_allow_html=True
        )
        st.button(
            "Evaluate Model", 
            key="action_btn_evaluate",
            use_container_width=True, 
            on_click=set_workspace_action, 
            args=("Evaluate",)
        )

    # Col 6: ACTIVE LEARNING
    with col6:
        st.markdown(
            """
            <div class='domain-card' style='height: 150px;'>
                <h3>🧠 Active Learn</h3>
                <p>Review only the model's most uncertain predictions.</p>
            </div>
            """, unsafe_allow_html=True
        )
        st.button(
            "Start Active Review", 
            key="action_btn_al", 
            use_container_width=True, 
            on_click=set_workspace_action,
            args=("Active_Learning",)
        )
        
    st.markdown("---")
    if st.button("← Back to Workspaces Home", key="back_from_action_choice"):
        st.session_state.current_workspace = None
        st.session_state.current_domain = None
        navigate_to_home()
        st.rerun()
# ==============================
# ANNOTATION PAGE (FINAL CORRECTED VERSION WITH DYNAMIC ENTITY EDITOR)
# ==============================
def show_annotation_page():
    if not st.session_state.logged_in_email or not st.session_state.current_workspace:
        navigate_to_home()
        st.rerun()
        return

    user_email = st.session_state.logged_in_email
    workspace_name = st.session_state.current_workspace
    domain = st.session_state.current_domain
    
    st.markdown(f"<div class='title'>📝 Annotate Data: <span style='color:#6EC6FF;'>{workspace_name}</span></div>", unsafe_allow_html=True)
    st.info("Annotate the intent and entities for each sentence. The predictions below are from the base spaCy NLU model and serve as suggestions.")

    # 1. Load Data
    if st.session_state.sentences_df is None:
        local_cursor_data = conn.cursor()
        # Fetch the last_modified time to use as a cache key for dataset loading
        local_cursor_data.execute(
            "SELECT last_modified FROM workspaces WHERE user_email=? AND workspace_name=?", 
            (user_email, workspace_name)
        )
        # Use a default time if not found, though should not happen here
        result = local_cursor_data.fetchone()
        last_modified = result[0] if result else "2000-01-01" 
        
        # Attempt to load the dataset
        dataset_df = load_dataset_blob(user_email, workspace_name, last_modified)
        
        if dataset_df is None:
            st.error("No dataset found for this workspace. Please go to **Upload & Train** to upload a CSV first.")
            if st.button("Go to Upload & Train"):
                set_workspace_action("Train")
            return

        # Generate sentences from the dataset
        st.session_state.sentences_df = split_dataframe_to_sentences(dataset_df)
        st.session_state.annotation_index = 0 # Reset index on new data load

    sentences_df = st.session_state.sentences_df
    total_sentences = len(sentences_df)

    # Check if annotation is complete
    if st.session_state.annotation_index >= total_sentences:
        st.balloons()
        st.success(f"🎉 Annotation Complete! You have labeled **{total_sentences}** sentences.")
        st.markdown("You can now go to **Upload & Train** to train your custom NLU model!")
        if st.button("Go to Train Data"):
            set_workspace_action("Train")
        st.markdown("---")
        if st.button("← Change Action", key="change_action_btn_annotate_done"):
            st.session_state.workspace_action = None
            navigate_to_action_choice()
            st.rerun()
        return

    # 2. Display the current sentence & Pre-load existing data
    current_index = st.session_state.annotation_index
    current_sentence = sentences_df.loc[current_index, 'sentence']
    
    # --- START ENTITY PRE-POPULATION LOGIC (ADAPTED FOR DYNAMIC EDITOR) ---
    existing_intent, existing_entities_json = get_existing_annotation(user_email, workspace_name, current_sentence)
    predicted_intent, predicted_entities_json = predict_intent_and_entities(current_sentence, domain)

    # 1. Prepare entities as lists of dicts (for the session state)
    existing_entities_list = entities_input_to_dict(json_to_simple_entities(existing_entities_json))
    predicted_entities_list = entities_input_to_dict(json_to_simple_entities(predicted_entities_json))
    
    # 2. Determine initial input values (Highest priority first)
    initial_intent_value = existing_intent if existing_intent else (predicted_intent if predicted_intent != "default_fallback" else "")
    
    # Entities List (List of dicts): Existing DB > Model Prediction > Empty List
    initial_entities_list = existing_entities_list if existing_entities_list else predicted_entities_list

    # CRITICAL: Reset the session state for the new sentence ONLY if the sentence index changed
    if st.session_state.get('last_annotation_index') != current_index:
        st.session_state.current_entities = initial_entities_list # Initialize the list of dicts for the editor
        st.session_state.last_annotation_index = current_index
    # --- END ENTITY PRE-POPULATION LOGIC ---

    st.progress(current_index / total_sentences, text=f"Progress: {current_index + 1}/{total_sentences} sentences to process.")
    
    st.markdown("### Sentence to Annotate:")
    st.markdown(f"<div class='sentence-display'>**{current_sentence}**</div>", unsafe_allow_html=True)
    
    # --- ANNOTATION FORM ---
    col_intent, col_predicted = st.columns([1, 1])
    
    with col_intent:
        st.markdown("### 1. Intent:")
        # Intent Input
        selected_intent = st.text_input(
            label="Intent Label", 
            value=initial_intent_value, 
            key=f"intent_input_{current_index}",
            placeholder="e.g., query_balance, book_flight"
        )

    with col_predicted:
        st.markdown("### Base Model Suggestions:")
        # Show predicted intent
        st.markdown(f"**Intent Suggestion:** `{predicted_intent}`")
        # Show predicted entities
        st.markdown(f"**Entity Suggestions:** `{json_to_simple_entities(predicted_entities_json)}`")

    st.markdown("---")
    
    # 3. Dynamic Entity Editor
    # The editor manages st.session_state.current_entities
    render_dynamic_entity_editor(current_index)

    st.markdown("---")
    
    # 4. Navigation/Save Buttons
    col_prev, col_save, col_skip = st.columns([1, 2, 1])

    with col_prev:
        if st.button("← Previous", use_container_width=True, key="prev_btn", disabled=current_index == 0):
            # Reset entity state before going back
            st.session_state.last_annotation_index = -1 
            st.session_state.annotation_index = max(0, current_index - 1)
            st.rerun()

    with col_save:
        if st.button("✅ Save & Next", use_container_width=True, type="primary", key="save_btn"):
            # Validation: checks if the intent text input is not empty
            if not selected_intent.strip():
                st.error("Please enter a valid Intent before saving.")
                return
            
            # --- SAVE LOGIC ADAPTED FOR DYNAMIC EDITOR ---
            # Entities are already in st.session_state.current_entities (list of dicts)
            entities_list = st.session_state.current_entities 
            # Convert the list of dicts to the required JSON string for the DB
            entities_json = dict_to_entities_json_string(entities_list) 
            
            if save_annotation_to_db(workspace_name, user_email, current_sentence, selected_intent.strip(), entities_json):
                st.session_state.annotation_index += 1
                st.toast(f"Saved: Intent='{selected_intent}'", icon='📝')
                st.rerun()
            # --- END SAVE LOGIC ADAPTED ---

    with col_skip:
        if st.button("→ Skip", use_container_width=True, key="skip_btn"):
            st.session_state.annotation_index += 1
            st.toast("Sentence skipped.", icon='⏭️')
            st.rerun()
            
    st.markdown("---")
    
    local_cursor = conn.cursor()
    local_cursor.execute("SELECT COUNT(*) FROM annotations WHERE workspace_name=?", (workspace_name,))
    total_labeled = local_cursor.fetchone()[0]
    st.info(f"**Total Labeled Examples Saved in DB:** {total_labeled}")

    if st.button("← Change Action", key="back_from_annotate"):
        st.session_state.workspace_action = None
        navigate_to_action_choice()
        st.rerun()
# ==============================
# ACTIVE LEARNING PAGE
# ==============================
def show_active_learning_page():
    if not st.session_state.logged_in_email or not st.session_state.current_workspace:
        navigate_to_home()
        st.rerun()
        return

    user_email = st.session_state.logged_in_email
    workspace_name = st.session_state.current_workspace
    domain = st.session_state.current_domain 
    
    st.markdown(f"<div class='title'>🧠 Active Learning Review: <span style='color:#6EC6FF;'>{workspace_name}</span></div>", unsafe_allow_html=True)
    st.info("Focusing annotation on **uncertain samples** where model confidence is below 60%. This saves time and improves quality faster.")
    st.markdown("---")
    
    # -----------------------------------------------------------
    # CRITICAL FIX: INTEGRATE UPLOAD IF DATASET IS MISSING
    # -----------------------------------------------------------
    if st.session_state.sentences_df is None:
        
        local_cursor_data = conn.cursor()
        local_cursor_data.execute(
            "SELECT last_modified FROM workspaces WHERE user_email=? AND workspace_name=?", 
            (user_email, workspace_name)
        )
        result = local_cursor_data.fetchone()
        last_modified = result[0] if result else "2000-01-01" 
        
        dataset_df = load_dataset_blob(user_email, workspace_name, last_modified)
        
        if dataset_df is None:
            st.subheader("⚠️ Dataset Missing: Upload Required")
            st.warning("Please upload a CSV file to begin Active Learning for this workspace.")
            
            uploaded_file = st.file_uploader(
                "Choose a CSV file containing your raw sentences.", 
                type=['csv'],
                key="csv_uploader_al_page"
            )
            
            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
                    df_preview = pd.read_csv(uploaded_file)
                    st.dataframe(df_preview.head())
                    
                    if st.button(f"Save '{uploaded_file.name}' and Start AL", key="save_dataset_al_btn", type="primary"):
                        uploaded_file.seek(0)
                        file_data_bytes = uploaded_file.getvalue()
                        
                        local_cursor_train = conn.cursor()
                        local_cursor_train.execute(
                            """
                            INSERT INTO datasets (user_email, workspace_name, filename, data) 
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(user_email, workspace_name) DO UPDATE SET 
                            filename = excluded.filename, 
                            data = excluded.data
                            """,
                            (user_email, workspace_name, uploaded_file.name, sqlite3.Binary(file_data_bytes))
                        )
                        local_cursor_train.execute("UPDATE workspaces SET last_modified=CURRENT_TIMESTAMP WHERE user_email=? AND workspace_name=?", (user_email, workspace_name))
                        conn.commit()
                        
                        st.session_state.sentences_df = split_dataframe_to_sentences(df_preview)
                        st.session_state.al_queue = None # Force queue regeneration
                        st.session_state.al_index = 0
                        st.success("✅ Dataset saved! Starting Active Learning.")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error processing or saving dataset: {e}")
            return # Exit function if data is missing or upload is ongoing

        # If data was loaded successfully from DB, split and proceed
        st.session_state.sentences_df = split_dataframe_to_sentences(dataset_df)
        st.session_state.al_queue = None # Force queue regeneration on first load
        st.session_state.al_index = 0
        st.rerun() # Rerun to proceed with the filled sentences_df
    # -----------------------------------------------------------
    
    # --- Now proceed with queue generation ---
    
    df_sentences = st.session_state.sentences_df
    if df_sentences.empty:
         st.error("The dataset contains zero sentences. Please upload a valid CSV.")
         return

    # --- 1. Queue Generation / Check ---
    if st.session_state.al_queue is None:
        st.subheader("Generating Uncertainty Queue...")
        
        queue_data = []

        try:
            with st.spinner(f"Scoring {len(df_sentences)} sentences to find uncertain samples..."):
                for index, row in df_sentences.iterrows():
                    sentence = row['sentence']
                    
                    # Call the function that runs SpaCy and simulation
                    intent, entities_json, confidence = predict_with_confidence(sentence, domain)
                    
                    if confidence < 0.60: # --- FILTER STEP: Threshold set at 60% ---
                        queue_data.append({
                            'sentence': sentence,
                            'predicted_intent': intent,
                            'predicted_entities_json': entities_json,
                            'confidence': confidence
                        })
                time.sleep(0.5)

            if not queue_data:
                st.success("🎉 All samples are highly confident! No uncertain data found for review.")
                st.session_state.al_queue = [] # Mark as empty
                return
            
            st.session_state.al_queue = queue_data
            st.session_state.al_index = 0
            st.rerun() # Rerun to display the queue
            
        except NameError as e:
            # Catches if predict_with_confidence or any dependency is misspelled/missing
            st.error(f"CRITICAL ERROR: Active Learning queue failed to generate due to a missing function definition ({e}).")
            return
        except Exception as e:
            # Catches general execution errors inside the loop
            st.error(f"CRITICAL ERROR: Active Learning queue generation failed unexpectedly: {e}")
            return

    # --- 2. Process Queue ---
    queue = st.session_state.al_queue
    index = st.session_state.al_index
    total_uncertain = len(queue)
    
    if index >= total_uncertain:
        st.success("✅ Active Learning queue complete! You have reviewed all uncertain samples.")
        st.session_state.al_queue = None
        return
        
    current_sample = queue[index]
    current_sentence = current_sample['sentence']
    
    # Display Progress
    st.progress(index / total_uncertain, text=f"Reviewing uncertain samples: {index + 1}/{total_uncertain}")
    
    st.markdown("### Sentence to Review:")
    st.markdown(f'<div class="sentence-display">**{current_sentence}**</div>', unsafe_allow_html=True)

    # --- Display Model Uncertainty ---
    st.error(f"**Model Confidence:** {current_sample['confidence']:.2f} (Uncertain!)")
    st.markdown("---")
    
    # --- Annotation Tools (Correction Form) ---
    with st.form("al_annotation_form"):
        col_intent, col_predicted = st.columns([1, 1])
        
        with col_intent:
            st.markdown("### 1. Correct Intent:")
            # Pre-fill intent with prediction as a starting point for correction
            corrected_intent = st.text_input(
                "Enter Correct Intent:",
                value=current_sample['predicted_intent'],
                key="al_corrected_intent"
            )

        with col_predicted:
            st.markdown("### Model Prediction:")
            st.markdown(f"**Predicted Intent:** `{current_sample['predicted_intent']}`")
            # Display Entities as string input for easy viewing/editing
            st.markdown(f"**Predicted Entities:** `{json_to_simple_entities(current_sample['predicted_entities_json'])}`")
            
        st.markdown("---")
        
        # Entities Input (Simplified Text Area for correction)
        st.markdown("### 2. Correct Entities (Type:Value, Type:Value...)")
        corrected_entities_input = st.text_input(
            "Enter corrected/verified entities (e.g., location:Paris, date:tomorrow)",
            value=json_to_simple_entities(current_sample['predicted_entities_json']),
            key="al_corrected_entities_input"
        )
        
        # 3. Save Button
        col_save, col_skip = st.columns([1, 1])
        
        with col_save:
            save_button = st.form_submit_button("✅ Save Correction & Next", type="primary", use_container_width=True)

        with col_skip:
            st.form_submit_button("⏭️ Skip & Next", type="secondary", use_container_width=True, on_click=lambda: st.session_state.update(al_index=index + 1))


        if save_button:
            # --- Save Logic (Similar to Annotation Page) ---
            if not corrected_intent.strip():
                st.error("Please enter a valid Intent before saving.")
                return

            # Convert string input back to JSON for DB 
            entities_dict = {}
            try:
                if corrected_entities_input.strip():
                    parts = corrected_entities_input.split(',')
                    for part in parts:
                        if ':' in part:
                            k, v = part.split(':', 1)
                            entities_dict[k.strip()] = v.strip()
                        else:
                            st.error("Error parsing entities. Ensure format is: `name:value, name:value`")
                            return
            except Exception:
                st.error("Error parsing entities. Ensure format is: `name:value, name:value`")
                return
            
            entities_json = json.dumps(entities_dict)

            # Use the existing utility to save the annotation
            if save_annotation_to_db(workspace_name, user_email, current_sentence, corrected_intent.strip(), entities_json):
                st.session_state.al_index += 1
                st.toast(f"Saved AL correction: {corrected_intent}", icon='🧠')
                st.rerun()


# ==============================
# EVALUATION PAGE (RE-ADDED)
# ==============================

def show_evaluation_page():
    if not st.session_state.logged_in_email or not st.session_state.current_workspace:
        navigate_to_home()
        st.rerun()
        return

    user_email = st.session_state.logged_in_email
    workspace_name = st.session_state.current_workspace
    
    st.markdown(f"<div class='title'>📊 Model Evaluation: <span style='color:#6EC6FF;'>{workspace_name}</span></div>", unsafe_allow_html=True)
    
    # --- Check for annotations/training data ---
    annotated_data = pd.read_sql("SELECT * FROM annotations WHERE user_email=? AND workspace_name=?", conn, params=(user_email, workspace_name))
    annotation_count = len(annotated_data)
    
    if annotation_count == 0:
        st.warning("No labeled data found for this workspace. Please **Annotate Data** and **Train** before evaluating.")
        if st.button("Go to Annotation"):
            set_workspace_action("Annotate")
        return
        
    st.success(f"Evaluation will run on a simulated test set derived from your {annotation_count} labeled examples.")
    st.markdown("---")

    # --- Use Tabs for clarity (MODIFIED TO INCLUDE TAB 4) ---
    tab1, tab2, tab3, tab4 = st.tabs(["Run New Evaluation", "History & Metrics", "Model Comparison (Local)", "Workspace Comparison (Global)"]) 

    # ----------------------------------------------------
    # TAB 1: RUN NEW EVALUATION (Unchanged)
    # ----------------------------------------------------
    with tab1:
        st.subheader("1. Run New Evaluation")
        st.info("A new 'version' of your NLU model (simulated) will be evaluated on test data.")
        
        col_v, col_n, col_b = st.columns([1.5, 3, 1])
        
        # ... (Form logic for running evaluation remains the same) ...
        
        with col_v:
            history_df = get_model_history(user_email, workspace_name)
            current_versions = len(history_df)
            default_version = f"v{current_versions + 1}"
            new_version = st.text_input("Model Version Name", value=default_version, key="new_model_version")
            
        with col_n:
            notes = st.text_input("Evaluation Notes", placeholder="e.g., Trained with 100 new examples", key="eval_notes")
            
        with col_b:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) # Spacer
            if st.button("🚀 Run Evaluation", key="run_evaluation_btn", type="primary", use_container_width=True):
                if new_version.strip():
                    # NOTE: Using the core training simulator function for consistency
                    acc, f1, samples = _simulate_nlu_training_core(user_email, workspace_name, new_version.strip(), "simple_model", 50, notes)
                    
                    if acc and f1:
                        st.success(f"✅ Evaluation complete for **{new_version}**!")
                        st.metric(label="Accuracy", value=f"{acc*100:.2f}%")
                        st.metric(label="F1 Score", value=f"{f1*100:.2f}%")
                        time.sleep(1) 
                        st.rerun() 
                    elif acc is None and f1 is None:
                        # Error message displayed by core function handles version conflict
                        pass
                else:
                    st.error("Please provide a version name.")
        


    # ----------------------------------------------------
    # TAB 2: HISTORY & METRICS (Comparison Table & Chart)
    # ----------------------------------------------------
    with tab2:
        # ... (Inside the 'with tab2:' block) ...
        
        st.markdown("### Metric Visualization")
        
        if not history_df.empty:
            # 1. Select the relevant numeric columns and reset the index
            chart_df = history_df[['Version', 'Accuracy', 'F1 Score']]
            
            # 2. Restructure the data to "long format" for plotting:
            chart_data_long = pd.melt(
                chart_df, 
                id_vars=['Version'], 
                value_vars=['Accuracy', 'F1 Score'],
                var_name='Metric', 
                value_name='Score'
            )
            
            # 3. Plot the data using st.bar_chart, instructing it how to group:
            try:
                import plotly.express as px
                
                # Create the grouped bar chart using Plotly Express
                fig = px.bar(
                    chart_data_long, 
                    x='Version', 
                    y='Score', 
                    color='Metric', 
                    barmode='group', # CRITICAL: This ensures bars are separated, not stacked.
                    text_auto='.2f',
                    title='Accuracy and F1 Score Comparison by Model Version'
                )
                fig.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig, use_container_width=True)
                
            except ImportError:
                st.warning("Install 'plotly' (`pip install plotly`) for advanced bar grouping. Falling back to simple chart.")
                # Fallback to the original stacked chart if Plotly is missing:
                chart_data_wide = history_df.set_index('Version')[['Accuracy', 'F1 Score']]
                st.bar_chart(chart_data_wide)

    # ----------------------------------------------------
    # TAB 3: MODEL COMPARISON (Confusion Matrix)
    # ----------------------------------------------------
    with tab3:
        st.subheader("3. Two-Model Comparison (Confusion Matrix)")
        
        history_df = get_model_history(user_email, workspace_name)
        model_options = ['-- Select Model --'] + history_df['Version'].tolist()

        if len(history_df) < 2:
            st.warning("You need at least two model versions saved to run a comparison.")
        else:
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                model1_version = st.selectbox("Select Model 1 (Baseline)", options=model_options, key="compare_model_1")
            
            with col_m2:
                model2_version = st.selectbox("Select Model 2 (Candidate)", options=model_options, key="compare_model_2")

            if model1_version != '-- Select Model --' and model2_version != '-- Select Model --':
                
                # Check for self-comparison
                if model1_version == model2_version:
                    st.warning("Please select two different model versions for comparison.")
                else:
                    st.markdown("---")
                    st.markdown(f"#### Comparison: **{model2_version}** vs. **{model1_version}**")
                    
                    # Call the function to display the confusion matrix
                    display_confusion_matrix(model1_version, model2_version, history_df)
                    
    # ----------------------------------------------------
    # TAB 4: WORKSPACE COMPARISON (Global - NEW CONTENT)
    # ----------------------------------------------------
    with tab4:
        show_workspace_comparison_tab()


    # --- Back Button (Remains the same) ---
    # Near the end of def show_evaluation_page():

 # --- Back Button (Remains the same) ---
    st.markdown("---")
    if st.button("← Change Action", key="back_from_evaluate_model"): # KEEP THIS ONE
        st.session_state.workspace_action = None
        navigate_to_action_choice()
        st.rerun()
    
                    
    

# ----------------------------------------------------
# New Function: SIMULATE AND DISPLAY CONFUSION MATRIX
# ----------------------------------------------------
def display_confusion_matrix(model1_version, model2_version, history_df):
    """
    Simulates and displays a confusion matrix for the two selected models 
    as a color-coded heatmap using dynamic intent labels from annotations.
    """
    user_email = st.session_state.logged_in_email 
    workspace_name = st.session_state.current_workspace
    
    # Inner helper function for simulation 
    def simulate_matrix(accuracy, intent_labels):
        num_labels = len(intent_labels)
        matrix = np.zeros((num_labels, num_labels), dtype=int)
        total_samples = 100 
        # Calculate samples that will be correctly placed on the diagonal
        correct_samples = int(total_samples * accuracy * 0.95) 
        incorrect_samples = total_samples - correct_samples
        
        # Fill the diagonal (Correct predictions)
        diag_value = int(correct_samples / num_labels)
        for i in range(num_labels):
            matrix[i, i] = diag_value
            
        # Distribute the remaining samples and errors
        remaining = correct_samples - (diag_value * num_labels)
        if num_labels > 0:
            matrix[0, 0] += remaining
        
        # Distribute the incorrect samples (Off-diagonal errors)
        for _ in range(incorrect_samples):
            if num_labels == 0: break
            true_idx = random.randint(0, num_labels - 1)
            pred_idx = random.randint(0, num_labels - 1)
            # Ensure predicted is not the true label (unless fully random distribution is desired)
            if pred_idx == true_idx:
                 pred_idx = (pred_idx + 1) % num_labels
                
            matrix[true_idx, pred_idx] += 1
            
        return matrix

    st.markdown("#### Simulated Confusion Matrix")
    
    # 1. Fetch metrics to influence simulation
    m1_row = history_df[history_df['Version'] == model1_version].iloc[0]
    m2_row = history_df[history_df['Version'] == model2_version].iloc[0]
    
    # Accuracy is used to weight the simulation
    m1_acc = m1_row['Accuracy']
    m2_acc = m2_row['Accuracy']
    
    # ----------------------------------------------------------------------
    # --- 2. Get a dynamic list of intents from the ANNOTATIONS table (FIX)---
    # ----------------------------------------------------------------------
    domain = st.session_state.current_domain

    # Fetch all unique intents present in the user's annotated data
    local_cursor = conn.cursor()
    local_cursor.execute(
        """
        SELECT DISTINCT intent FROM annotations 
        WHERE user_email = ? AND workspace_name = ?
        """,
        (user_email, workspace_name)
    )
    
    # Convert the result list of tuples to a simple list of strings
    dynamic_intents = [row[0] for row in local_cursor.fetchall()]

    if not dynamic_intents:
        # Fallback to domain default intents or generic ones if no custom data exists
        intents = DOMAINS.get(domain, {}).get("intents", ["Intent_A", "Intent_B", "Intent_C", "Intent_D"])
    else:
        # Use the dynamic list of intents
        intents = dynamic_intents
    
    # Limit to a maximum of 4 intents for clear visualization/simulation capacity
    intent_labels = intents[:4]
    num_labels = len(intent_labels)

    if num_labels == 0:
        st.error("Cannot generate confusion matrix: No valid intents found in annotations.")
        return

    # 3. Simulate Matrices
    matrix1 = simulate_matrix(m1_acc, intent_labels)
    matrix2 = simulate_matrix(m2_acc, intent_labels)
    
    # Create DataFrames with Intent Labels
    cm_df1 = pd.DataFrame(matrix1, index=intent_labels, columns=intent_labels)
    cm_df2 = pd.DataFrame(matrix2, index=intent_labels, columns=intent_labels)

    # 4. Display Matrices Side-by-Side with HEATMAP STYLING
    col_disp1, col_disp2 = st.columns(2)
    
    with col_disp1:
        st.markdown(f"##### {model1_version} (Acc: {m1_acc:.2f})")
        
        # Apply background gradient for the visual heatmap effect
        st.dataframe(
            cm_df1.style.background_gradient(cmap='Blues', axis=None),
            use_container_width=True
        )
        st.caption(f"Rows = **True Intent** (Actual Label); Columns = **Predicted Intent**.")

    with col_disp2:
        st.markdown(f"##### {model2_version} (Acc: {m2_acc:.2f})")
        
        # Apply background gradient for the visual heatmap effect
        st.dataframe(
            cm_df2.style.background_gradient(cmap='Blues', axis=None),
            use_container_width=True
        )
        st.caption(f"Rows = **True Intent** (Actual Label); Columns = **Predicted Intent**.")            
    st.markdown("---")

# ==============================
# WORKSPACE / CHAT PAGE (RESTRICTED BY ACTION)
# ==============================
def show_workspace_page():
    # --- START OF EXISTING SETUP ---
    # NOTE: This function assumes 'navigate_to_home', 'DOMAINS', 
    # 'set_workspace_action', 'is_model_trained', 'show_chat_test_page', 
    # 'show_training_page', 'show_annotation_page', and 'show_evaluation_page' 
    # are all defined elsewhere in your script.
    
    if not st.session_state.logged_in_email or not st.session_state.current_workspace or not st.session_state.workspace_action:
        navigate_to_home()
        st.rerun()
        return

    user_email = st.session_state.logged_in_email
    workspace_name = st.session_state.current_workspace
    domain = st.session_state.current_domain
    action = st.session_state.workspace_action
    domain_display = DOMAINS.get(domain, {}).get("icon", "❓") + " " + domain
    
    st.markdown(f"<div class='title'>Action: <span style='color:#6EC6FF;'>{action}</span> for {workspace_name}</div>", unsafe_allow_html=True)
    # --- END OF EXISTING SETUP ---
    
    
    # =========================================================================
    # Action Handling: Call the appropriate page function based on the action
    # =========================================================================

    if action == "Test":
        st.subheader(f"💬 Chat & Test - {domain_display} Domain")
        # Check if the model has been trained (simulated check)
        if is_model_trained(workspace_name):
            # Assumes show_chat_test_page is defined elsewhere
            show_chat_test_page(workspace_name, user_email) 
        else:
            st.warning("Please **Train** your NLU model first before testing.")
            if st.button("Go to Train Data", key="go_to_train_from_test_page"):
                set_workspace_action("Train")
                st.rerun()

    elif action == "Train":
        # *** CORRECT IMPLEMENTATION: Call the dedicated training page function ***
        show_training_page() 
        
    elif action == "Annotate":
        st.subheader(f"🏷️ Annotate Data - {domain_display} Domain")
        try:
            # Assumes show_annotation_page is defined elsewhere
            show_annotation_page()
        except NameError:
            st.error("Function 'show_annotation_page()' is not defined.")
            
    # 🚨 ADD THIS BLOCK TO HANDLE ACTIVE LEARNING 🚨
    elif action == "Active_Learning": # Assuming your action is named this way
        st.subheader(f"🧠 Active Learning - {domain_display} Domain")
        try:
            # The actual Active Learning page logic
            show_active_learning_page() 
        except NameError:
            st.error("Function 'show_active_learning_page()' is not defined.")

    elif action == "Feedback":
        st.subheader(f"📝 Feedback Review - {domain_display} Domain")
        try:
        # Assumes show_feedback_page() is defined and ready
            show_feedback_page()
        except NameError:
            st.error("Function 'show_feedback_page()' is not defined or callable.")

    elif action == "Evaluate":
        st.subheader(f"📊 Evaluate Model - {domain_display} Domain")
        try:
            # Assumes show_evaluation_page is defined elsewhere
            show_evaluation_page()
        except NameError:
            st.error("Function 'show_evaluation_page()' is not defined.")
            
    else:
        # Catch unexpected actions, providing a button to return to action center
        st.error(f"Invalid action selected: '{action}'. Returning to Action Center.")
        st.button("Return to Action Center", on_click=navigate_to_action_choice)


    # --- Back Buttons (Keep existing back logic) ---
    st.markdown("---")
    if st.button("← Back to Workspaces Home", key="back_to_home_workspace"):
        st.session_state.current_workspace = None
        st.session_state.current_domain = None
        st.session_state.workspace_action = None
        st.session_state.messages = []
        navigate_to_home()
        st.rerun()
# ==============================
# AUTH & REGISTRATION PAGES
# ==============================
def show_register_page():
    st.markdown("<div class='title'>Welcome to Sign Up <span style='color:#6EC6FF;'>Buddy!</span></div>", unsafe_allow_html=True)
    with st.form(key="register_form"):
        name = st.text_input("Enter your name", key="reg_name")
        email = st.text_input("Enter your email", key="reg_email")
        password = st.text_input("Enter your password", type="password", key="reg_password")
        st.markdown(
            "By signing up, you agree to our <a href='?page=policy'>Terms and Privacy Policy</a>.",
            unsafe_allow_html=True
        )
        agree = st.checkbox("I confirm I have read and agree to the policy.", key="register_agree_checkbox")
        
        if st.form_submit_button("Sign Up", type="primary", use_container_width=True):
            local_cursor = conn.cursor()
            if name and email and password and agree:
                hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                try:
                    local_cursor.execute(
                        "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                        (name, email, hashed_pw)
                    )
                    conn.commit()
                    st.success("Registration successful! Please log in.")
                    navigate_to_login()
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("This email is already registered. Please log in.")
            else:
                st.error("Please fill in all fields and agree to the policy.")
    
    if st.button("Already have an account? Log In"):
        navigate_to_login()
        st.rerun()

def show_login_page():
    st.markdown("<div class='title'>Welcome Back, <span style='color:#6EC6FF;'>Buddy!</span></div>", unsafe_allow_html=True)
    with st.form(key="login_form"):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.form_submit_button("Login", type="primary", use_container_width=True):
            local_cursor = conn.cursor()
            local_cursor.execute("SELECT password FROM users WHERE email=?", (email,))
            result = local_cursor.fetchone()

            if result and bcrypt.checkpw(password.encode('utf-8'), result[0]):
                st.session_state.logged_in_email = email
                st.success("Login successful!")
                
                # Log the successful login activity
                log_activity(email, "LOGIN", "Successful login from UI.")
                
                # --- NEW ADMIN NAVIGATION LOGIC ---
                admin_email = "admin@example.com" # Define admin email locally
                if email == admin_email:
                    navigate_to_admin() # Direct admin to admin portal
                else:
                    navigate_to_home() # Regular users go to home
                # --- END NEW ADMIN NAVIGATION LOGIC ---
                
                st.rerun()
            else:
                st.error("Invalid email or password.")
                
    if st.button("Don't have an account? Register"):
        navigate_to_register()
        st.rerun()

def show_policy_page():
    st.markdown("<div class='title'>Privacy Policy</div>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background-color: #1c2b4d; padding: 20px; border-radius: 10px; border: 1px solid #334466;'>
        <p>This policy outlines how your information is handled within the BuddyBot application.</p>
        <h4>Data Collection</h4>
        <p>We collect personal information you provide directly to us, suchs as your name, email address, and interactions with the 'Buddy' AI.</p>
        <h4>Data Usage</h4>
        <p>We use your data to operate, maintain, and improve our services. We do not sell your personal data to third parties.</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("← Back to Registration", key="back_to_reg"):
        navigate_to_register()
        st.rerun()


# ==============================
# PAGE LAYOUT EXECUTION
# ==============================

# CALL SIDEBAR ONCE HERE - IT WILL RUN ON EVERY PAGE EXCEPT 'register'
# ==============================
# PAGE LAYOUT EXECUTION
# ==============================
# CALL SIDEBAR ONCE HERE - IT WILL RUN ON EVERY PAGE EXCEPT 'register'
show_sidebar_content()

# Reroute all pages that are set by the action-setting callbacks (1-to-1 page routing)
if st.session_state.page == 'workspace':
    show_workspace_page()
elif st.session_state.page == 'annotate': 
    show_annotation_page()
elif st.session_state.page == 'feedback_module': # ADD THIS LINE
    show_feedback_page()
elif st.session_state.page == 'active_learning': 
    show_active_learning_page()
elif st.session_state.page == 'admin': # <-- This calls the new function
    show_admin_portal()
elif st.session_state.page == 'training':
    show_training_page()
elif st.session_state.page == 'evaluate':
    show_evaluation_page()
elif st.session_state.page == 'home':
    show_home_page()
elif st.session_state.page == 'create_workspace':
    show_create_workspace_page()
elif st.session_state.page == 'action_choice':
    # CRITICAL FIX: This is the missing piece that ensures the Action Choice page loads.
    show_action_choice_page() 
else:
    # Landing page for Register/Login/Policy (Split Screen Implementation)
    
    # Only show the split screen for login/register
    if st.session_state.page in ('login', 'register'):
        
        # Use two columns for the split screen effect (1.2:1 ratio)
        col1, col2 = st.columns([1.2, 1])

        with col1:
            # Decorative Elements (Left Side) - Style adjusted for better alignment
            st.markdown(
                """
                <div style='max-width: 450px; margin-left: 50px; padding-top: 50px;'>
                <div class='logo-container' style='text-align: left; margin-top: 0;'>
                    <img src='https://cdn-icons-png.flaticon.com/512/4712/4712100.png' width='150'>
                </div>
                <div class='chat-bubble-container' style='text-align: left;'>
                    <div class='chat-bubble'>Hello, can you help me?</div><br>
                    <div class='chat-bubble'>Of course! Buddy is ready to assist.</div>
                </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        with col2:
            # Authentication Forms (Right Side)
            st.markdown("<div style='padding-top: 50px;'></div>", unsafe_allow_html=True) # Add padding to align forms
            if st.session_state.page == 'register':
                # Assuming you have a function called show_register_page()
                show_register_page()
            elif st.session_state.page == 'login':
                # Assuming you have a function called show_login_page()
                show_login_page()
    
    # Handle the Policy page separately if it's not a split-screen layout
    elif st.session_state.page == 'policy':
        # Assuming you have a function called show_policy_page()
        show_policy_page()
        